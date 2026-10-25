import os
os.environ["USE_TRITON_FP8"] = "0"

import json
import time
import numpy as np
from my_utils import MultiModalAgent, extract_answer
from dataset import load_data, create_question
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="Slake", type=str)
    parser.add_argument("--mode", default="test", type=str)
    parser.add_argument("--answer", default="MC", type=str)
    parser.add_argument("--size", default=200, type=int)
    parser.add_argument("--model", default="4B-Instruct", type=str)
    parser.add_argument("--save_name", default=None, type=str)
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--lora", default=None, type=str)
    
    args = parser.parse_args()
    router = MultiModalAgent(f"Qwen/Qwen3-VL-{args.model}", lora=args.lora)
    final_acc = []
    final_res = []
    start_time = time.time()
    test_data = load_data(args.dataset, args.mode)
    for r in range(args.repeat):
        cnt, acc, extracted_answers = 0, [], {}
        for idx, data in enumerate(test_data):
            question, image_path, label, answer_type = create_question(data, args.dataset, args.mode, args.answer)
            if question is None: continue
            
            response = router.chat(question+"Answer in the format: <think>Your reason<answer>Your answer</answer>.", image_path)
            answer = extract_answer(response, answer_type)
            cnt += 1
            print(f"Finish {r}-{cnt} with {answer} and label {label}")
            extracted_answers[cnt] = {
                "repeat": r,
                "index": idx,
                "question": question,
                "response": response,
                "answer": answer,
                "label": label
            }
            if args.answer == "MC":
                correct = 1 if answer == label else 0
                acc.append(correct)
            if cnt == args.size: break
        if args.answer == "MC":
            final_acc.append(sum(acc)/len(acc))
        if cnt > 0:
            final_res.append(extracted_answers)
    save_name = args.save_name.lower() if args.save_name else args.model.lower()
    save_dir = os.path.join("./log", save_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.dataset}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_res, f, indent=2, ensure_ascii=False)
    print(f"results are saved at: {save_path}")
    if args.answer == "MC":
        print(f"Finish evaluation with accuracy mean {np.mean(final_acc)} and std {np.std(final_acc)} on {args.model}/{args.dataset}")
    end_time=time.time()
    print(f"time used {end_time-start_time} seconds")
