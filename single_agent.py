import asyncio
from dotenv import load_dotenv
from agents import Runner

from diff_agents.singleAgent import single_agent
from data.loader import (
    load_nejm,
    format_single_agent_question,
    get_ground_truth,
    image_to_base64,
)
from evaluation.evaluator_old import Evaluator

load_dotenv()

N_SAMPLES = 689
SEED = 42
CONDITION = "single_agent_baseline"
MODEL_NAME = "qwen2.5vl:7b"  


async def run_pipeline(example: dict) -> str:
    """Single agent sees clinical text, options, and image in one forward pass."""
    question_text = format_single_agent_question(example)
    base64_image = image_to_base64(example["image"])

    agent_input = [
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
                    "text": question_text
                }
            ]
        }
    ]

    result = await Runner.run(single_agent, agent_input)
    return result.final_output


async def main():
    print(f"Loading {N_SAMPLES} NEJM cases (seed={SEED})...")
    dataset = load_nejm(n_samples=N_SAMPLES, seed=SEED)

    evaluator = Evaluator(
        run_name=f"{CONDITION}-{MODEL_NAME}-n{N_SAMPLES}-seed{SEED}",
        config={
            "condition": CONDITION,
            "model": MODEL_NAME,
            "dataset": "nejm-image-challenge",
            "n_samples": N_SAMPLES,
            "seed": SEED,
        }
    )

    for example in dataset:
        print(f"\nCase {example['image_id']}: {example['question'][:80]}...")
        agent_output = await run_pipeline(example)
        evaluator.log_single_agent_example(
            image_id=example["image_id"],
            question=example["question"],
            agent_output=agent_output,
            ground_truth=get_ground_truth(example),
        )

    evaluator.finish()


if __name__ == "__main__":
    asyncio.run(main())