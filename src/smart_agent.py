import os
import re
import json
import time
import numpy as np
from dataset import load_data, create_question
from argparse import ArgumentParser
from my_utils import MultiModalAgent, extract_answer, extract_new_answer, MultiAgentAggregation

system_prompt = "You are a professional and experienced medical expert. Please analyze the given image to answer the question."
def template(question, answer_type):
    return f"""
Analyze the provided medical image and answer the following {answer_type} question: <image> {question}

## Instructions
1. **Visual Analysis**: Thoroughly examine the image and identify all anatomical structures, abnormalities, or features relevant to the question.
2. **Evidence-Based Reasoning**: Base your conclusion strictly on the visual evidence. Cite specific observations that support your analysis.
3. **Uncertainty Annotation**: If the evidence is genuinely ambiguous or critical features are obscured, return UNCERTAIN in your response.
## Output Format
<reason>[Your visual analysis citing specific evidence]</reason><certainty>[CERTAIN/UNCERTAIN]</certainty><answer>[Your definitive best-effort answer: {answer_type}]</answer>
·
## Examples

**High Confidence Case:**
<reason>The CT slice clearly shows a hyperdense, well-circumscribed lesion in the right lung base with characteristics typical of a hamartoma.</reason><certainty>CERTAIN</certainty><answer>Benign</answer>

**Low Confidence Case:**
<reason>The image shows a possible mass in the mediastinum, but the slice level appears to be above the cardiac silhouette and the structure boundaries are poorly defined due to motion artifact.</reason><certainty>UNCERTAIN</certainty><answer>Malignant</answer>

## Critical Rules
1. The `<certainty>` tag MUST be either `CERTAIN` or `UNCERTAIN` (no other values allowed).
2. The `<answer>` tag MUST always contain a specific {answer_type} value, regardless of certainty level.
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="Slake", type=str)
    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--answer", default="MC", type=str)
    parser.add_argument("--lora", default=None, type=str)
    parser.add_argument("--save_name", default="smart_med", type=str)
    parser.add_argument("--size", default=200, type=int)
    parser.add_argument("--repeat", default=3, type=int)
    args = parser.parse_args()
    print("Loading lora from: ", args.lora)
    specialist = MultiModalAgent("Qwen/Qwen3-VL-4B-Instruct", lora=args.lora)
    final_acc, final_answer = [], []
    start_time = time.time()
    test_data = load_data(args.dataset, args.mode)
    for r in range(args.repeat):
        acc, extracted_answers, cnt = [], {}, 0
        for idx, data in enumerate(test_data):
            question, image_path, label, answer_type = create_question(data, args.dataset, args.mode, args.answer)
            if question is None: continue
            cnt += 1
            response = None
            user = template(question, answer_type)
            initial_response = specialist.chat(user, image_path, system_prompt=system_prompt)
            answer, certainty = extract_new_answer(initial_response, answer_type)
            print(f"{[cnt]} Initial Assessment: ", answer, certainty)
            if certainty and certainty.upper() == "UNCERTAIN":
                router_response = specialist.chat(question, image_path, role="router")
                specialists = [s.strip() for s in re.findall(r"<specialist>(.*?)</specialist>", router_response, re.DOTALL) if s.strip()]
                if specialists:
                    parsing_result = (specialists, ["Analyze the given medical image"] * len(specialists))
                else:
                    print(f"Parsing error: {router_response}")
                    parsing_result = (["Medical Imaging Expert"], ["Analyze the given medical image"])
                print(f"[{cnt}] Experts: ", parsing_result)
                expert, role_descriptions = parsing_result
                syn_report, response, expert_answers = MultiAgentAggregation(specialist, question, image_path, expert, answer_type, role_descriptions=role_descriptions, initial_assessment=initial_response)
                answer = extract_answer(response, answer_type)
                certainty = "UNCERTAIN"
                print(f"[{cnt}] Expert answers {expert_answers} | Final answers {answer}")
            correct = 1 if answer.upper() == label else 0
            acc.append(correct)
            extracted_answers[cnt] = {
                "repeat": r,
                "idx": idx,
                "question": question,
                "response": initial_response if not response else response,
                "image": image_path,
                "label": label,
                "answer": answer
            }
            print(f"[{cnt}] Pred: {answer} | Label: {label} | Correct: {correct}")
            if cnt == args.size: break
        final_acc.append(sum(acc)/len(acc))
        final_answer.append(extracted_answers)
    end_time=time.time()
    save_dir = os.path.join("./log", args.save_name.lower())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.dataset}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
    print(f"results are saved at: {save_path}")
    print("Accuracy: ", final_acc, np.mean(final_acc), np.std(final_acc))
    print(f"Finish evaluation on {args.dataset} with {args.size} samples x {args.repeat} repeats: "
          f"accuracy mean {np.mean(final_acc):.4f}, std {np.std(final_acc):.4f}")
    print(f"time used {end_time-start_time} seconds")
    print("Lora path: ", args.lora)
