"""
Output-only multi-agent pipeline on NEJM Image Challenge.

Runs text agent + vision agent + meta agent with output-level communication.
Logs comprehensive per-case metrics aligned with ThoughtComm for paired analysis.

Usage:
    # Full dataset (689 cases) — for 3-way comparison (output vs CoT vs structured)
    python main.py

    # Test split only (339 cases) — for 4-way comparison including ThoughtComm
    python main.py --split test

    # Quick smoke test
    python main.py --n 10
"""

import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv
from agents import Runner

from diff_agents.textAgent import text_agent
from diff_agents.visionAgent import vision_agent
from diff_agents.metaAgent import meta_agent
from data.loader import (
    load_nejm,
    format_text_question,
    format_vision_question,
    format_meta_question,
    get_ground_truth,
    image_to_base64,
)
from evaluation.evaluator import Evaluator

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

COMMUNICATION_MODE = "output_only"
SEED = 42
TRAIN_SPLIT = 350  # must match ThoughtComm's split


# ── Pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(example: dict) -> tuple[str, str, str]:
    """
    Output-only communication with distributed multimodal input:
    - Text agent sees full clinical context + options (no image)
    - Vision agent sees diagnostic question + options + image (no patient history)
    - Meta agent sees both final outputs + original case + options
    """

    # --- Text Agent (no image) ---
    text_question = format_text_question(example)
    result_text = await Runner.run(text_agent, text_question)
    text_output = result_text.final_output

    # --- Vision Agent (image + question only) ---
    base64_image = image_to_base64(example["image"])
    vision_question = format_vision_question(example)

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

    # --- Meta Agent (output-only: sees both final answers) ---
    meta_input = format_meta_question(example, text_output, vision_output)
    result_meta = await Runner.run(meta_agent, meta_input)
    meta_output = result_meta.final_output

    return text_output, vision_output, meta_output


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Output-only multi-agent NEJM evaluation")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all",
                        help="Dataset split: 'all' for full 689, 'test' for ThoughtComm-aligned 339")
    parser.add_argument("--n", type=int, default=None,
                        help="Override sample count (for smoke tests)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (must be 42 to align with ThoughtComm)")
    return parser.parse_args()


async def main():
    args = parse_args()

    # Load dataset with split support
    all_cases = load_nejm(n_samples=None, seed=args.seed)

    if args.split == "test":
        dataset = all_cases[TRAIN_SPLIT:]
        split_label = f"test_{len(dataset)}"
    elif args.split == "train":
        dataset = all_cases[:TRAIN_SPLIT]
        split_label = f"train_{len(dataset)}"
    else:
        dataset = all_cases
        split_label = f"all_{len(dataset)}"

    # Override with --n if provided (for smoke tests)
    if args.n is not None:
        dataset = dataset[:args.n]
        split_label = f"{split_label}_n{args.n}"

    n_cases = len(dataset)
    print(f"Running {COMMUNICATION_MODE} on {n_cases} NEJM cases (split={args.split}, seed={args.seed})")

    evaluator = Evaluator(
        run_name=f"{COMMUNICATION_MODE}_{split_label}_seed{args.seed}",
        config={
            "communication_mode": COMMUNICATION_MODE,
            "model": "medgemma1.5:4b",
            "meta_model": "medgemma1.5:4b",
            "dataset": "NEJM",
            "split": args.split,
            "n_samples": n_cases,
            "seed": args.seed,
            "train_split": TRAIN_SPLIT,
        }
    )

    for i, example in enumerate(dataset):
        print(f"\n[{i+1}/{n_cases}] Case {example['image_id']:04d}: "
              f"{example['question'][:80]}...")

        text_output, vision_output, meta_output = await run_pipeline(example)

        evaluator.log_multi_agent(
            image_id=example["image_id"],
            question=example["question"],
            ground_truth=get_ground_truth(example),
            text_agent_output=text_output,
            vision_agent_output=vision_output,
            meta_output=meta_output,
            brier_score=example.get("brier_score", 0.0),
        )

    evaluator.finish()


if __name__ == "__main__":
    asyncio.run(main())