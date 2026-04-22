import os
import asyncio
from dotenv import load_dotenv
from agents import Runner

from diff_agents.textAgent import text_agent
from diff_agents.visionAgent import vision_agent
from diff_agents.metaAgent import meta_agent
from data.loader import load_vqarad, format_text_question, format_vision_question, get_ground_truth, image_to_base64
from evaluation.evaluator import Evaluator

load_dotenv()

N_SAMPLES = 100
COMMUNICATION_MODE = "output_only"

async def run_pipeline(example: dict) -> tuple[str, str, str]:
    """
    Output-only communication with distributed multimodal input:
    - Text agent sees question text only
    - Vision agent sees image + question
    - Meta agent combines both outputs
    """

    # --- Text Agent (no image) ---
    text_question = format_text_question(example)
    result_text = await Runner.run(text_agent, text_question)
    text_output = result_text.final_output

    # --- Vision Agent (image + question) ---
    base64_image = image_to_base64(example["image"])
    vision_question = format_vision_question(example)

    # Pass image as multimodal content
    vision_input = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": vision_question
                }
            ]
        }
    ]

    result_vision = await Runner.run(vision_agent, vision_input)
    vision_output = result_vision.final_output

    # --- Meta Agent (output-only: sees both final answers, no reasoning traces) ---
    meta_input = f"""You are reviewing assessments from two specialists for this question:

Question: {example["question"]}

Clinical Text Specialist Assessment:
{text_output}

Radiology Vision Specialist Assessment:
{vision_output}

Synthesise both assessments and provide your final answer."""

    result_meta = await Runner.run(meta_agent, meta_input)
    meta_output = result_meta.final_output

    return text_output, vision_output, meta_output


async def main():
    print(f"Loading {N_SAMPLES} VQA-RAD examples (closed questions only)...")
    dataset = load_vqarad(split="test", n_samples=N_SAMPLES)

    evaluator = Evaluator(
    run_name=f"{COMMUNICATION_MODE}100samples-llava7b-llama31-8b",
    config={
        "communication_mode": COMMUNICATION_MODE,
        "text_model": "llama3.1:8b",
        "vision_model": "llava:7b",
        "dataset": "vqa-rad",
        "n_samples": N_SAMPLES,
    }
)

    for i, example in enumerate(dataset):
        print(f"\nExample {i+1}/{N_SAMPLES}: {example['question']}")

        text_output, vision_output, meta_output = await run_pipeline(example)

        evaluator.log_example(
            question=example["question"],
            text_agent_output=text_output,
            vision_agent_output=vision_output,
            meta_output=meta_output,
            ground_truth=get_ground_truth(example),
            example_idx=i
        )

    evaluator.finish()


if __name__ == "__main__":
    asyncio.run(main())