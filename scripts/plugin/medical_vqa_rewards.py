import collections
import logging
import os
import re
from typing import Any, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:  # available at training time; the fallback keeps the module importable
    from swift.rewards import ORM, orms
except Exception:  # noqa: BLE001
    orms = {}

    class ORM:  # type: ignore[no-redef]
        def __init__(self, args=None, **kwargs):
            self.args = args

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset", "MedVR-3K")

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CERTAINTY_TAG = re.compile(r"<certainty>(.*?)</certainty>", re.DOTALL)

_PREFIX = re.compile(
    r"^(?:the\s+)?(?:final\s+|definitive\s+|best\s+)?"
    r"(?:answer|option|choice|ans)\s*(?:is|was|:|=)?\s*",
    re.IGNORECASE)
_LEADING_LETTER = re.compile(r"^([A-D])\s*(?:[:.)\],;、，]|\s|$)")

_PAYOFF = {
    ("correct", "CERTAIN"): 1.0,
    ("correct", "UNCERTAIN"): 0.8,
    ("correct", None): 0.25,
    ("incorrect", "CERTAIN"): -0.5,
    ("incorrect", "UNCERTAIN"): 0.25,
    ("incorrect", None): -0.25,
}

_FORMAT_BONUS = 0.05
_MAX_DUP_PENALTY = 0.10

_COUNTERS: collections.Counter = collections.Counter()


def _rank0() -> bool:
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception: 
        return True


def _log(msg: str, *args) -> None:
    if not _rank0():
        return
    print("[rewards_v2] " + (msg % args if args else msg), flush=True)


def _resolve_image_path(images, index):
    """Resolve ms-swift image entries or raw relative dataset paths."""
    if not images or index >= len(images):
        return None
    image = images[index]
    if isinstance(image, (list, tuple)):
        image = image[0] if image else None
    if isinstance(image, dict):
        image = image.get("path") or image.get("image") or image.get("url")
    if not isinstance(image, str):
        return None
    if os.path.isabs(image) or os.path.exists(image):
        return image
    candidate = os.path.join(DATASET_ROOT, image)
    return candidate if os.path.exists(candidate) else image


def extract_answer(text: str) -> Optional[str]:
    """Normalised answer: last <answer> tag, markdown/prefix/option text stripped.

    Same name as in v1 so external callers keep working; the semantics are
    stricter (normalised) rather than looser.
    """
    if not text:
        return None
    matches = _ANSWER_TAG.findall(text)
    if not matches:
        return None
    raw = matches[-1]  # last tag wins: an early draft inside <reason> must not
    s = raw.strip().strip("*_`\"' \t\r\n")
    s = re.sub(r"[\s*_`。.!]+$", "", s).strip()
    s = _PREFIX.sub("", s).strip()
    s = s.strip("*_`\"' \t\r\n.,;:!()[]{}，。；：、】")
    if not s:
        return None
    upper = s.upper()
    if upper in {"YES", "NO"}:
        return upper
    if upper in {"Y", "N"}:
        return {"Y": "YES", "N": "NO"}[upper]
    if re.fullmatch(r"[A-D]", upper):
        return upper
    m = _LEADING_LETTER.match(upper)
    if m:
        return m.group(1)
    tokens = re.findall(r"\b(YES|NO)\b", upper)
    if tokens:
        return tokens[0]
    return upper.rstrip(".")


def parse_certainty(text: str) -> Optional[str]:
    """Return CERTAIN / UNCERTAIN / None (missing or ambiguous/invalid).

    ``None`` is "the model did not commit to a level" and is scored through the
    MISSING column of the payoff table.
    """
    if not text:
        return None
    matches = _CERTAINTY_TAG.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip().upper()
    if not raw:
        return None
    has_certain = re.search(r"\bCERTAIN\b", raw) is not None
    has_uncertain = re.search(r"\bUNCERTAIN\b", raw) is not None
    if has_certain and has_uncertain:
        # e.g. the prompt placeholder "<certainty>CERTAIN/UNCERTAIN</certainty>"
        return None
    if has_uncertain or "UNCERT" in raw:
        return "UNCERTAIN"
    if has_certain or "CERT" in raw:
        return "CERTAIN"
    return None


def extract_certainty(text: str) -> str:
    """v1-compatible signature; missing/invalid maps to UNCERTAIN."""
    return parse_certainty(text) or "UNCERTAIN"


def _well_formed(text: str, tag: str) -> bool:
    if text.count(f"<{tag}>") != 1 or text.count(f"</{tag}>") != 1:
        return False
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return bool(m and m.group(1).strip())


def _flush_counters() -> None:
    if _rank0() and _COUNTERS:
        _log("cert_missing=%d cert_invalid=%d answer_missing=%d",
             _COUNTERS["certainty_missing"], _COUNTERS["certainty_invalid"],
             _COUNTERS["answer_missing"])
        _COUNTERS.clear()


class MedicalVQAAccuracy(ORM):
    """
    Certainty-aware accuracy reward.
    """
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        missing_policy = os.environ.get("CERTAINTY_MISSING", "penalize").strip().lower()
        rewards: List[float] = []
        for completion, sol in zip(completions, solution):
            pred = extract_answer(completion)
            gt = extract_answer(sol)
            certainty = parse_certainty(completion)

            if gt is None:
                rewards.append(0.0)          # unparseable gold: nothing to score
                continue
            if pred is None:
                _COUNTERS["answer_missing"] += 1
                rewards.append(_PAYOFF[("incorrect", None)])
                continue
            if certainty is None:
                if completion and "<certainty>" in completion:
                    _COUNTERS["certainty_invalid"] += 1
                else:
                    _COUNTERS["certainty_missing"] += 1
                if missing_policy == "pessimistic":
                    certainty = "CERTAIN"
                else:
                    rewards.append(
                        _PAYOFF[("correct" if pred == gt else "incorrect", None)])
                    continue
            rewards.append(_PAYOFF[("correct" if pred == gt else "incorrect", certainty)])

        _flush_counters()
        return rewards


class MedicalVQAFormat(ORM):
    """
    Format reward
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            text = completion or ""
            reason_ok = _well_formed(text, "reason")
            answer_ok = _well_formed(text, "answer")
            certainty_ok = parse_certainty(text) is not None

            duplicated = sum(max(0, text.count(f"<{t}>") - 1)
                             for t in ("reason", "answer", "certainty"))
            score = _FORMAT_BONUS * (reason_ok + answer_ok + certainty_ok)
            score -= _FORMAT_BONUS * min(duplicated, int(_MAX_DUP_PENALTY // _FORMAT_BONUS))
            if not answer_ok:
                score -= _FORMAT_BONUS
            rewards.append(max(-0.15, min(0.15, score)))
        return rewards


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

JUDGE_PROMPT_TEMPLATE = """### User's Question
[{question}]

### Assistant Response
[{response}]

## Evaluation Rubrics

Evaluate the assistant's response on three dimensions, each scored 1-5:
1. Logic Consistency: coherence, structure, and validity of the reasoning chain.
2. Factual Correctness: medical accuracy of the response.
3. Image Alignment: how well the response utilizes and correctly interprets visual information.

## Output Format
<answer>score_1|score_2|score_3</answer>
"""

# Must stay identical to `JUDGE_SYSTEM_PROMPT` in scripts/build_judge_data.py.
JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for multimodal medical QA tasks. Your role "
    "is to critically assess the assistant's response to the provided User "
    "Input."
)

FORCED_PREFIX = "<answer>"
SCORE_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)

class MedVLMReward(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        import torch

        from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        base_model = os.environ.get("VLM_JUDGE_BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
        lora_path = os.environ.get("VLM_JUDGE_LORA", "")

        self.processor = AutoProcessor.from_pretrained(base_model)
        self.processor.tokenizer.padding_side = "right"
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": torch.cuda.current_device()}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **model_kwargs)
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = next(self.model.parameters()).device

        # --- score-token layout, validated once (fail fast, not mid-run) ---
        tok = self.processor.tokenizer
        self.digit_ids = []
        for value in SCORE_VALUES:
            ids = tok.encode(str(int(value)), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"score token {int(value)} is not a single token ({ids}); the "
                    "logits-based judge requires single-token digits.")
            self.digit_ids.append(ids[0])
        self._digit_ids_t = torch.tensor(self.digit_ids, dtype=torch.long, device=self.device)
        self._values_t = torch.tensor(SCORE_VALUES, dtype=torch.float32, device=self.device)
        self._digit_value = dict(zip(self.digit_ids, SCORE_VALUES))
        pipe_ids = tok.encode("|", add_special_tokens=False)
        if len(pipe_ids) != 1:
            raise RuntimeError(f"score separator '|' is not a single token: {pipe_ids}")
        self._pipe_id = pipe_ids[0]
        self.max_new_tokens = max(6, int(os.environ.get("JUDGE_MAX_NEW_TOKENS", "8")))
        self._score_fallbacks = 0

        self.w_logic = 0.3
        self.w_fact = 0.5
        self.w_img = 0.2

        self.batch_size = max(1, int(os.environ.get("JUDGE_BATCH_SIZE", "4")))
        self._fallback_cfg = os.environ.get("JUDGE_FALLBACK", "ema").strip().lower()
        self._ema: Optional[float] = None
        self._ema_decay = 0.9
        self._load_image_failures = 0

    # -- text / image prep ------------------------------------------------ #

    def _build_text(self, prompt: str, with_image: bool) -> str:
        content: List[Any] = []
        if with_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return text + FORCED_PREFIX

    def _load_image(self, path) -> Optional[Any]:
        if path is None:
            return None
        try:
            from PIL import Image
            return Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            self._load_image_failures += 1
            logger.warning("[rewards_v2] cannot open judge image %s: %s", path, exc)
            return None

    # -- scoring ---------------------------------------------------------- #

    def _value_from_logits(self, logits_row) -> float:
        """Expected score over {1..5} at one position (renormalised over digits)."""
        probs = self.torch.softmax(logits_row[self._digit_ids_t].float(), dim=-1)
        return float((probs * self._values_t).sum().item())

    def _combine(self, dims: Sequence[float]) -> float:
        s_logic, s_fact, s_img = [(d - 1.0) / 4.0 for d in dims]  # 1..5 -> 0..1
        reward = self.w_logic * s_logic + self.w_fact * s_fact + self.w_img * s_img
        return float(np.clip(reward, 0.0, 1.0))

    def _locate_scores(self, tokens: Sequence[int], scores, row: int):
        n = len(tokens)
        for start in range(n - 4):
            if (tokens[start] in self._digit_value
                    and tokens[start + 1] == self._pipe_id
                    and tokens[start + 2] in self._digit_value
                    and tokens[start + 3] == self._pipe_id
                    and tokens[start + 4] in self._digit_value):
                return [self._value_from_logits(scores[start][row]),
                        self._value_from_logits(scores[start + 2][row]),
                        self._value_from_logits(scores[start + 4][row])]
        return None

    def _score_autoregressive(self, prompts: Sequence[str], images, with_image: bool):
        torch = self.torch
        tokenizer = self.processor.tokenizer
        texts = [self._build_text(p, with_image) for p in prompts]

        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # generation needs left padding
        try:
            proc_kwargs: dict = dict(text=texts, return_tensors="pt", padding=True)
            if with_image:
                proc_kwargs["images"] = list(images)
            inputs = self.processor(**proc_kwargs).to(self.device)
        finally:
            tokenizer.padding_side = previous_side

        prompt_len = inputs["input_ids"].shape[1]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=pad_id)

        new_tokens = generated.sequences[:, prompt_len:]
        texts = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        scores = generated.scores  # tuple(step) of (B, V) logits
        out: List[Optional[float]] = []
        for i in range(len(texts)):
            dims = self._locate_scores([int(t) for t in new_tokens[i]], scores, i)
            out.append(self._combine(dims) if dims else None)
        return out

    def _score_group(self, prompts: Sequence[str], images: Optional[Sequence[Any]],
                     with_image: bool) -> List[Optional[float]]:
        try:
            out = self._score_autoregressive(prompts, images, with_image)
        except Exception as exc:  # noqa: BLE001
            _log("judge generate failed (%d samples): %s", len(prompts), exc)
            out = [None] * len(prompts)
        self._score_fallbacks += sum(1 for r in out if r is None)
        return out

    def _fallback_value(self) -> float:
        if self._fallback_cfg == "ema":
            return 0.5 if self._ema is None else self._ema
        try:
            return float(self._fallback_cfg)
        except ValueError:
            return 0.5

    def _update_ema(self, values: Sequence[float]) -> None:
        if not values:
            return
        batch_mean = float(np.mean(values))
        self._ema = (batch_mean if self._ema is None
                     else self._ema_decay * self._ema + (1 - self._ema_decay) * batch_mean)

    # -- entry point ------------------------------------------------------ #

    def __call__(self, completions: List[str], question=None, images=None,
                 **kwargs) -> List[float]:
        n = len(completions)
        if isinstance(question, (list, tuple)):
            questions = list(question)
        else:
            questions = [question or ""] * n
        if len(questions) < n:
            questions += [""] * (n - len(questions))

        paths = [_resolve_image_path(images, i) for i in range(n)]
        pil_images = [self._load_image(p) for p in paths]
        prompts = [JUDGE_PROMPT_TEMPLATE.format(question=questions[i],
                                                response=completions[i])
                   for i in range(n)]

        rewards: List[Optional[float]] = [None] * n
        failures = 0
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            chunk = list(range(start, stop))
            for with_image in (True, False):
                group = [i for i in chunk
                         if (pil_images[i] is not None) == with_image]
                if not group:
                    continue
                try:
                    scores = self._score_group(
                        [prompts[i] for i in group],
                        [pil_images[i] for i in group] if with_image else None,
                        with_image)
                except Exception as exc:  # noqa: BLE001
                    _log("judge batch failed (%d samples): %s", len(group), exc)
                    scores = [None] * len(group)
                for i, score in zip(group, scores):
                    rewards[i] = score
            self._update_ema([r for r in rewards[start:stop] if r is not None])
            failures += sum(1 for r in rewards[start:stop] if r is None)

        fallback = self._fallback_value()
        final = [fallback if r is None else float(r) for r in rewards]

        if _rank0():
            valid = [r for r in rewards if r is not None]
            if valid:
                summary = (f"mean={np.mean(valid):.4f} min={min(valid):.4f} "
                           f"max={max(valid):.4f} std={np.std(valid):.4f}")
            else:
                summary = "mean=n/a"
            _log("judge n=%d ok=%d fail=%d fallback=%d img_fail=%d %s ema=%s",
                 n, len(valid), failures, self._score_fallbacks,
                 self._load_image_failures, summary,
                 f"{self._ema:.4f}" if self._ema is not None else "n/a")
            self._load_image_failures = 0
            self._score_fallbacks = 0
        return final


if orms:
    orms["medical_vqa_acc"] = MedicalVQAAccuracy
    orms["medical_vqa_format"] = MedicalVQAFormat
    orms["medical_vqa_judge"] = MedVLMReward
