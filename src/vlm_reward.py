import importlib.util
import json
import os
import sys
from argparse import ArgumentParser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_PATH = os.path.join(REPO_ROOT, "scripts", "plugin", "medical_vqa_rewards.py")
DEFAULT_IMAGE_ROOT = os.path.join(REPO_ROOT, "dataset", "MedVR-3K")

_MODULE_NAME = "smartmed_judge_rewards"


def _load_plugin(path=None):
    """Import the GRPO reward module by file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path or PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class VLMJudgeScorer:
    def __init__(self, lora=None, base=None, gpu=None, image_root=None,
                 batch_size=None, verbose=False, plugin_path=None):
        # Set before the plugin imports torch (it does so in __init__).
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if lora is not None:
            os.environ["VLM_JUDGE_LORA"] = lora
        if base:
            os.environ["VLM_JUDGE_BASE_MODEL"] = base
        if batch_size:
            os.environ["JUDGE_BATCH_SIZE"] = str(batch_size)

        self.image_root = image_root or DEFAULT_IMAGE_ROOT
        self.plugin = _load_plugin(plugin_path)
        if not verbose:
            # The training path logs one "[rewards_v2] judge n=..." line per call.
            self.plugin._log = lambda *a, **k: None

        self._dims = []
        dims = self._dims
        parent = self.plugin.MedVLMReward

        class _RecordingJudge(parent):
            """Records the dimension scores the parent is about to combine."""

            def _combine(self, values):
                dims.append([float(v) for v in values])
                return super()._combine(values)

        self.engine = _RecordingJudge()

    # -- paths ------------------------------------------------------------- #
    def _resolve_image(self, image):
        if not image:
            return None
        if os.path.isabs(image) or os.path.exists(image):
            return image
        candidate = os.path.join(self.image_root, image.lstrip("./"))
        return candidate if os.path.exists(candidate) else image

    # -- scoring ----------------------------------------------------------- #
    def score(self, question, response, image=None):
        """Score one response. Returns {dims, reward, scorable}."""
        if response is None:
            response = ""
        prompt = self.plugin.JUDGE_PROMPT_TEMPLATE.format(
            question=question or "", response=response)

        path = self._resolve_image(image)
        pil_image = self.engine._load_image(path) if path else None
        with_image = pil_image is not None

        self._dims.clear()
        values = self.engine._score_autoregressive(
            [prompt], [pil_image] if with_image else None, with_image)
        reward = values[0]
        if reward is None:
            return {"dims": None, "reward": None, "scorable": False}
        return {"dims": self._dims[-1], "reward": float(reward), "scorable": True}

    def score_batch(self, rows, progress=False):
        out = []
        for i, row in enumerate(rows):
            result = self.score(row.get("question"), row.get("response"),
                                row.get("image"))
            result["index"] = i
            if "id" in row:
                result["id"] = row["id"]
            out.append(result)
            if progress:
                print(f"\r[vlm_reward] {i + 1}/{len(rows)}", end="", file=sys.stderr,
                      flush=True)
        if progress:
            print(file=sys.stderr)
        return out


def main():
    parser = ArgumentParser(description="Score responses with the SMART-Med judge.")
    parser.add_argument("--lora", default=None,
                        help=f"judge LoRA checkpoint (default: None)")
    parser.add_argument("--base", default=None, help="judge base model")
    parser.add_argument("--gpu", default=None, help="physical GPU id")
    parser.add_argument("--image-root", default=None,
                        help=f"root for relative image paths (default: {DEFAULT_IMAGE_ROOT})")
    parser.add_argument("--question", default=None)
    parser.add_argument("--response", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--input", default=None, help="jsonl with question/response/image")
    parser.add_argument("--output", default=None, help="write jsonl here (batch mode)")
    parser.add_argument("--plugin", default=None,
                        help=f"reward module to import (default: {PLUGIN_PATH})")
    parser.add_argument("--verbose", action="store_true",
                        help="keep the plugin's per-call diagnostic lines")
    args = parser.parse_args()

    if args.input is None and args.question is None:
        parser.error("give --input <rows.jsonl> or --question/--response/--image")

    print(f"[vlm_reward] judge lora: {args.lora}")
    scorer = VLMJudgeScorer(lora=args.lora, base=args.base, gpu=args.gpu,
                            image_root=args.image_root, verbose=args.verbose,
                            plugin_path=args.plugin)

    if args.input:
        rows = []
        with open(args.input, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        results = scorer.score_batch(rows, progress=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fout:
                for result in results:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[vlm_reward] wrote {args.output} ({len(results)} rows)")
        else:
            for result in results:
                print(json.dumps(result, ensure_ascii=False))
        return

    result = scorer.score(args.question, args.response, args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
