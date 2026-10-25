import os
import json
import time
import numpy as np
from dataset import load_data, create_question
from argparse import ArgumentParser
from my_utils import MultiModalAgent, extract_answer, MultiAgentAggregation


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="Slake", type=str)
    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--answer", default="MC", type=str)
    parser.add_argument("--size", default=200, type=int)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--lora", default=None, type=str)
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
            parsing_result = (["Medical Imaging Expert", "Medical Imaging Expert", "Medical Imaging Expert"], ["Analyze the given medical image", "Analyze the given medical image", "Analyze the given medical image"])
            print(f"[{cnt}] Experts: ", parsing_result)
            expert, role_descriptions = parsing_result
            syn_report, response, expert_answers = MultiAgentAggregation(specialist, question, image_path, expert, answer_type, role_descriptions=role_descriptions)
            answer  = extract_answer(response, answer_type)
            print(f"[{cnt}] Expert answers {expert_answers} | Final answers {answer}")
            correct = 1 if answer.upper() == label else 0
            acc.append(correct)
            extracted_answers[cnt] = {
                "repeat": r,
                "idx": idx,
                "question": question,
                "response": response,
                "image": image_path,
                "label": label,
                "answer": answer
            }
            print(f"[{cnt}] Pred: {answer} | Label: {label} | Correct: {correct}")
            if cnt == args.size: break
        final_acc.append(sum(acc)/len(acc))
        final_answer.append(extracted_answers)
    end_time=time.time()
    save_path = f"./log/dense_agent/{args.dataset}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
    print(f"results are saved at: {save_path}")
    print("Accuracy: ", final_acc, np.mean(final_acc), np.std(final_acc))
    print(f"Finish evaluation on {args.dataset} with {args.size} samples")
    print(f"time used {end_time-start_time} seconds")
    print("Lora path: ", args.lora)
