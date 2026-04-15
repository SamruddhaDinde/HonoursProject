from datasets import load_dataset
import os


def load_medqa(split: str="test", n_samples: int = 10):
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        #"med_qa_en_source",
        #trust_remote_code=True,
        token =os.getenv("HF_TOKEN")
    )

    data = ds[split].select(range(n_samples))
    return data

def format_question(example: dict) -> str:
    options_str = "\n".join([f"{k}:{v}" for k, v in example["options"].items()])
    return f"""Question: {example["question"]} Options: {options_str}"""

def get_ground_truth(example:dict)-> str:
    return example["answer_idx"]



