import re
import json
import textwrap
import torch
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def _patch_vl_model(module):
    import types
    if hasattr(module, 'get_placeholder_mask'):
        def patched_get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
            if input_ids is None:
                return None, None
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            return special_image_mask, special_video_mask
        module.get_placeholder_mask = types.MethodType(patched_get_placeholder_mask, module)
    for name, child in module.named_children():
        _patch_vl_model(child)

class MultiModalAgent:
    def __init__(self, ckpt="Qwen/Qwen3-VL-2B-Instruct", system_prompt=None, lora=None):
        # Load the base model identically with or without LoRA, so the two
        # settings differ only by the adapter.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            ckpt, dtype="auto", device_map="cuda:0", attn_implementation="sdpa")
        _patch_vl_model(model.model)
        if lora:
            print("Loading LoRA adapter...")
            self.agent = PeftModel.from_pretrained(model, lora, torch_dtype="auto")
            try:
                self.processor = AutoProcessor.from_pretrained(lora)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(ckpt)
        else:
            self.agent = model
            self.processor = AutoProcessor.from_pretrained(ckpt)
        self.system_prompt = "You are a professional and experienced medical expert. Please analyze the given image to answer the question." if system_prompt == None else system_prompt
    
    def chat(self, question, image, system_prompt=None, role=None, expert=None, report=None, role_description=None, answer_type=None, initial_assessment=None):
        max_new_tokens = 512
        if role == "router":
            text = apply_router_template(question)
            system_prompt = "You are a medical resource allocation router. Your task is to allocate the medical question to the most appropriate specialist agents efficiently and effectively."
        elif role == "specialist":
            system_prompt = f"Your role is {expert}.\n"
            if role_description and role_description != "":
                system_prompt += f"Your task: {role_description}."
            system_prompt += "\n Please analyze the given image to answer the question."
            text = apply_specialist_template(question, answer_type=answer_type)
            max_new_tokens = 1024
        elif role == "moderator":
            system_prompt = "You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts."
            text = apply_moderator_template(question, report=report, answer_type=answer_type, initial_assessment=initial_assessment)
            max_new_tokens = 1024
        else:
            text = question
        messages = [
            {"role": "system","content": [{"type": "text","text": self.system_prompt if system_prompt is None else system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}
        ]

        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)
        device = next(self.agent.parameters()).device
        processed_inputs = {}
        for k, v in inputs.items():
            if k in ('pixel_values', 'image_grid_thw'):
                processed_inputs[k] = v.to(device) if hasattr(v, 'to') else v
            elif isinstance(v, list):
                if v and isinstance(v[0], list):
                    processed_inputs[k] = torch.tensor(v, dtype=torch.long, device=device)
                else:
                    processed_inputs[k] = torch.tensor([v[0]] if v else [0], dtype=torch.long, device=device)
            elif hasattr(v, 'to'):
                processed_inputs[k] = v.to(device)
        inputs = processed_inputs
        generated_ids = self.agent.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text

def apply_router_template(question):
    return f"""Based on the given medical question, determine which specialist can best answer this question: {question}.\n\nPlease choose from the following specialists: Radiology, Pathology, Oncology, Dermatology, Ophthalmology, Cardiology, Neurology, Pediatrics, Emergency Medicine, General Practice. Return the answer as: <specialist>SPECIALIST NAME</specialist>."""

def apply_specialist_template(question, answer_type):
    return f"""Analyze the provided image and answer this {answer_type} question: <image> {question}.\n## Instructions\n1. You should thoroughly scan the image and identify all visual elements relevant to the question (e.g., organs shown in the image).\n2. Your analysis must ground on the **evidence** you identified from the image.\n3. If the evidence is insufficient to draw a definite conclusion, respond **To Be Confirmed**. 4. Please return the final answer only {answer_type}/To Be Confirmed. \n## Output Format\n<think>YOUR ANALYSIS</think><answer>YOUR ANSWER</answer>\n## Example\n<think>I'm not 100% sure for my analysis to this medical image and the given question.</think><answer>To Be Confirmed</answer>"""

def apply_moderator_template(question, report, answer_type, initial_assessment=None):
    return f"""You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.
Given question: {question}
Initial assessment: {initial_assessment if initial_assessment else 'None'}
Expert reports: {report}
Please synthesize all expert opinions and provide a final answer in format: <answer>YOUR FINAL ANSWER</answer>. The final answer must be a specific {answer_type} value; do not use alternative labels such as TRUE, FALSE, CORRECT, or INCORRECT."""

def extract_answer(text, answer_type):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip().upper()
    return text.strip().upper() if text else ""


def extract_new_answer(text, answer_type):
    answer_pattern = r"<answer>(.*?)</answer>"
    certainty_pattern = r"<certainty>(.*?)</certainty>"
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    certainty_match = re.search(certainty_pattern, text, re.DOTALL)
    if answer_match and certainty_match:
        return answer_match.group(1).strip(), certainty_match.group(1).strip()
    return "N/A", "UNCERTAIN"


def MultiAgentAggregation(agent, question, image, expert, answer_type,
                          role_descriptions=None, initial_assessment=None):
    """Specialist agents analyze the case, then a moderator synthesizes the answer."""
    completions, expert_reports, expert_answers = [], [], []
    for i in range(len(expert)):
        role_desc = role_descriptions[i] if role_descriptions and i < len(role_descriptions) else None
        completion = agent.chat(
            question, image, role="specialist", expert=expert[i],
            role_description=role_desc, answer_type=answer_type)
        expert_answers.append(extract_answer(completion, answer_type))
        completions.append(completion)
    for count, (e, r, c) in enumerate(zip(expert, role_descriptions, completions), 1):
        expert_reports.append(textwrap.dedent(f"""{count}. {e} ({r}): {c}""").strip())
    syn_report = "\n\n".join(expert_reports)
    moderator_response = agent.chat(
        question, image, role="moderator", report=syn_report,
        answer_type=answer_type, initial_assessment=initial_assessment)
    return syn_report, moderator_response, expert_answers