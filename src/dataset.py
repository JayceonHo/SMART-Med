import json
import os

DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))

def load_data(dataset, mode="test"):
    path = f'{DATA_ROOT}/{dataset}/{mode}.jsonl'
    qa = []
    with open(path, 'r') as file:
        for line in file:
            qa.append(json.loads(line))
    return qa

def create_question(sample, dataset, mode="test", answer="MC"):
    if mode not in ["test", "train"]:
        question = sample["question"]
        img_path = sample["image_path"]
        label = sample["label"]
        answer_type = "free-form"
        return question, img_path, label, answer_type
    question, img_path, label = None, None, None
    answer_type = None
    root = DATA_ROOT
    if dataset == "PathQA": # only yes/no question can be exactly evalauted
        if answer == "MC" and sample["answer_type"] == "yes/no":
            question = sample["sent"]
            answer_type = "YES/NO"
        elif answer != "MC" and sample["answer_type"] != "yes/no":
            question = sample["sent"]
            answer_type = "Free-form"
        img_path = f"{root}/PathQA/raw_data/images/{mode}/" + sample["img_id"] + ".jpg"
        label = list(sample["label"].keys())[0].upper()
    elif dataset == "VQA-RAD":
        if answer=="MC" and sample["answer_type"] == "CLOSED" and sample["answer"].upper() in ["YES", "NO"]:
            question = sample["question"]
            answer_type = "YES/NO"
        elif answer != "MC" and sample["answer"].upper() not in ["YES", "NO"]:
            question = sample["question"]
            answer_type = "Free-form"
        img_path = f"{root}/VQA-RAD/images/" + sample["image_name"]
        label = sample["answer"].upper()
    elif dataset == "Slake":
        if answer=="MC" and sample["answer_type"] == "CLOSED" and sample["base_type"] == "vqa" and sample["answer"].upper() in ("YES", "NO"):
            question = sample["question"]
            answer_type = "YES/NO"
        elif answer != "MC" and sample["answer"].upper() not in ("YES", "NO"):
            question = sample["question"]
            answer_type = "Free-form"
        img_path = f"{root}/Slake/raw_data/imgs/" + sample["img_name"]
        label = sample["answer"].upper()
    elif dataset == "PMC-VQA":
        question = sample["question"]
        img_id = sample["img_id"]
        img_path = img_id if os.path.isabs(img_id) else f"{root}/PMC-VQA/images/" + img_id
        if not os.path.exists(img_path):
            basename = os.path.basename(img_id)
            img_path = next(
                (c for c in (
                    os.path.join(root, "PMC-VQA", "images", basename),
                    os.path.join(root, "MedVR-3K", "imgs", basename),
                ) if os.path.exists(c)),
                None,
            )
        if img_path is None:
            question = None
        label = sample["label"].upper()
        answer_type = "A/B/C/D"
    return question, img_path, label, answer_type