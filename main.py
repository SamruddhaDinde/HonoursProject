import os
import asyncio
from dotenv import load_dotenv
from agents import Runner

from diff_agents.textAgent import text_agent
from diff_agents.metaAgent import meta_agent
from data.loader import load_medqa, format_question, get_ground_truth
from evaluation.evaluator import Evaluator

load_dotenv()

MODEL = "llama3.1:8b"
N_SAMPLES = 10  # keep small while testing
COMMUNICATION_MODE = "output_only"

async def run_pipeline(question: str) -> tuple[str, str]:
    """
    Output-only communication:
    Agent A answers independently.
    Meta agent sees only Agent A's final answer, not its reasoning.
    """
    

    # Agent A answers
    result_a = await Runner.run(text_agent, question)
    agent_a_output = result_a.final_output

    # Meta agent gets the question + Agent A's output only
    meta_input = f"""You are reviewing one specialist's assessment of a medical question.

Original Question:
{question}

Specialist Assessment:
{agent_a_output}

Based on this assessment, provide your final answer."""

    result_meta = await Runner.run(meta_agent, meta_input)
    meta_output = result_meta.final_output

    return agent_a_output, meta_output

async def main():
    print(f"Loading {N_SAMPLES} MedQA examples...")
    dataset = load_medqa(split="test", n_samples=N_SAMPLES)

    evaluator = Evaluator(
        run_name=f"{COMMUNICATION_MODE}-{MODEL}",
        config={
            "communication_mode": COMMUNICATION_MODE,
            "model": MODEL,
            "dataset": "medqa",
            "n_samples": N_SAMPLES,
        }
    )

    for i, example in enumerate(dataset):
        question = format_question(example)
        ground_truth = get_ground_truth(example)

        print(f"\nExample {i+1}/{N_SAMPLES}")
        print(f"Ground truth: {ground_truth}")

        agent_a_output, meta_output = await run_pipeline(question)

        evaluator.log_example(
            question=question,
            agent_a_output=agent_a_output,
            meta_output=meta_output,
            ground_truth=ground_truth,
            example_idx=i
        )

    evaluator.finish()

if __name__ == "__main__":
    asyncio.run(main())