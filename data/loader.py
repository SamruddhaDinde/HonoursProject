# from datasets import load_dataset
# import os


# def load_medqa(split: str="test", n_samples: int = 10):
#     ds = load_dataset(
#         "GBaker/MedQA-USMLE-4-options",
#         #"med_qa_en_source",
#         #trust_remote_code=True,
#         token =os.getenv("HF_TOKEN")
#     )

#     data = ds[split].select(range(n_samples))
#     return data

# def format_question(example: dict) -> str:
#     options_str = "\n".join([f"{k}:{v}" for k, v in example["options"].items()])
#     return f"""Question: {example["question"]} Options: {options_str}"""

# def get_ground_truth(example:dict)-> str:
#     return example["answer_idx"]

import os
import base64
from io import BytesIO
from datasets import load_dataset

def load_vqarad(split: str = "test", n_samples: int = 10):
    """
    Load VQA-RAD dataset.
    Columns: image, question, answer (yes/no)
    """
    ds = load_dataset(
        "flaviagiammarino/vqa-rad",
        token=os.getenv("HF_TOKEN")
    )

    data = ds[split]

    if n_samples:
        data = data.select(range(min(n_samples, len(data))))

    return data

def image_to_base64(image) -> str:
    """Convert a PIL image to base64 string for vision agent."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def format_text_question(example: dict) -> str:
    """Text agent only sees the question — no image."""
    return f"""Question: {example["question"]}

Answer with YES or NO only, then explain your reasoning.
Respond in this exact format:
ANSWER: <YES or NO>
REASONING: <your clinical reasoning in 2-3 sentences>"""

def format_vision_question(example: dict) -> str:
    """Vision agent gets the question alongside the image."""
    return f"""You are analysing a medical image to answer a clinical question.

Question: {example["question"]}

Based on what you observe in the image, answer with YES or NO.
Respond in this exact format:
ANSWER: <YES or NO>
REASONING: <what you observed in the image that led to this answer>"""

def get_ground_truth(example: dict) -> str:
    """Returns the correct answer in uppercase."""
    return example["answer"].strip().upper()


