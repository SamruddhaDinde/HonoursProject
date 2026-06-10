"""
RAG-Enhanced Multi-Agent Pipeline on NEJM Image Challenge.

Same pipeline as Mode 1 (CoT single-round), but the text agent receives
retrieved medical knowledge from StatPearls alongside the clinical case.

The vision agent is UNCHANGED — it still sees only image + diagnostic question.
The meta agent is UNCHANGED — it still sees only specialist outputs + options.
The ONLY difference is that the text agent's prompt now includes relevant
medical textbook excerpts retrieved via RAG.

This isolates the effect of RAG: if accuracy improves, it's because the
text agent made better use of medical knowledge, not because the architecture
or communication mechanism changed.

Usage:
    python main_rag.py --split test     # 339 test cases
    python main_rag.py                   # all 689 cases
    python main_rag.py --n 5            # smoke test
"""

import os
import asyncio
import argparse
from dotenv import load_dotenv
from agents import Runner

from diff_agents.textAgent import text_agent
from diff_agents.visionAgent import vision_agent
from diff_agents.metaAgent import meta_agent
from data.loader import (
    load_nejm,
    format_vision_question,
    format_meta_question,
    get_ground_truth,
    image_to_base64,
    _format_options,
)
from evaluation.evaluator import Evaluator, extract_answer, is_correct
from rag.retriever import MedicalRetriever

import wandb

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

COMMUNICATION_MODE = "cot_single_round_with_rag"
SEED = 42
TRAIN_SPLIT = 350
RAG_TOP_K = 3


# ── RAG-enhanced text agent prompt ───────────────────────────────────────

def format_text_question_rag(example: dict, rag_context: str) -> str:
    """Text agent input with RAG: clinical case + retrieved medical knowledge + options.

    The retrieved context provides relevant medical textbook information
    that may help the text agent reason about the case. The agent is
    instructed to use this as reference, not as the answer.
    """
    if rag_context:
        rag_section = f"""
Reference Knowledge (from medical textbooks):
{rag_context}

Use the above reference knowledge to inform your reasoning, but base your
diagnosis on the specific clinical case details below.
"""
    else:
        rag_section = ""

    return f"""{rag_section}Clinical case:
{example["question"]}

Options:
{_format_options(example["options"])}

Based on the clinical history and any relevant reference knowledge, choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter A-E>
REASONING: <your clinical reasoning in 2-3 sentences>"""


# ── Pipeline ─────────────────────────────────────────────────────────────

async def run_rag_pipeline(example: dict, retriever: MedicalRetriever) -> dict:
    """
    Mode 1 + RAG: CoT single-round with RAG-enhanced text agent.

    1. Retrieve medical knowledge relevant to the clinical case
    2. Text agent reasons with clinical case + retrieved knowledge
    3. Vision agent reasons from image (unchanged)
    4. Meta agent synthesises both outputs (unchanged)
    """

    # ── RAG retrieval ─────────────────────────────────────────────────
    passages = retriever.retrieve(example["question"], top_k=RAG_TOP_K)
    rag_context = retriever.format_context(passages)

    # ── Text Agent (with RAG context) ─────────────────────────────────
    text_question = format_text_question_rag(example, rag_context)
    result_text = await Runner.run(text_agent, text_question)
    text_output = result_text.final_output

    # ── Vision Agent (unchanged) ──────────────────────────────────────
    base64_image = image_to_base64(example["image"])
    vision_question = format_vision_question(example)

    vision_input = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
                {"type": "text", "text": vision_question}
            ]
        }
    ]

    result_vision = await Runner.run(vision_agent, vision_input)
    vision_output = result_vision.final_output

    # ── Meta Agent (unchanged) ────────────────────────────────────────
    meta_input = format_meta_question(example, text_output, vision_output)
    result_meta = await Runner.run(meta_agent, meta_input)
    meta_output = result_meta.final_output

    return {
        "text_output": text_output,
        "vision_output": vision_output,
        "meta_output": meta_output,
        "rag_context": rag_context,
        "rag_passages": passages,
        "num_passages": len(passages),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RAG-enhanced multi-agent NEJM evaluation")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K,
                        help="Number of passages to retrieve per case")
    return parser.parse_args()


async def main():
    args = parse_args()

    # Load dataset
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

    if args.n is not None:
        dataset = dataset[:args.n]
        split_label = f"{split_label}_n{args.n}"

    n_cases = len(dataset)
    top_k = args.top_k

    print(f"Running {COMMUNICATION_MODE} on {n_cases} NEJM cases "
          f"(split={args.split}, seed={args.seed}, rag_top_k={top_k})")

    # ── Load RAG retriever ────────────────────────────────────────────
    print("\nLoading RAG index...")
    retriever = MedicalRetriever(top_k=top_k)

    # ── W&B init ──────────────────────────────────────────────────────
    evaluator = Evaluator(
        run_name=f"{COMMUNICATION_MODE}_{split_label}_seed{args.seed}_topk{top_k}",
        config={
            "communication_mode": COMMUNICATION_MODE,
            "model": "medgemma1.5:4b",
            "meta_model": "medgemma1.5:4b",
            "dataset": "NEJM",
            "split": args.split,
            "n_samples": n_cases,
            "seed": args.seed,
            "train_split": TRAIN_SPLIT,
            "rag_enabled": True,
            "rag_top_k": top_k,
            "rag_corpus": "MedRAG/statpearls",
            "rag_embedding": "nomic-embed-text",
        }
    )

    # ── Track RAG-specific metrics ────────────────────────────────────
    total_passages_retrieved = 0
    empty_retrievals = 0

    for i, case in enumerate(dataset):
        print(f"\n[{i+1}/{n_cases}] Case {case['image_id']:04d}: "
              f"{case['question'][:80]}...")

        try:
            result = await run_rag_pipeline(case, retriever)
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            continue

        # Track RAG stats
        total_passages_retrieved += result["num_passages"]
        if result["num_passages"] == 0:
            empty_retrievals += 1
            print(f"  WARNING: No passages retrieved for this case")

        # Log retrieved context titles
        if result["rag_passages"]:
            titles = [p["title"] for p in result["rag_passages"]]
            print(f"  RAG: {result['num_passages']} passages — {', '.join(titles[:2])}...")

        # Standard multi-agent logging
        evaluator.log_multi_agent(
            image_id=case["image_id"],
            question=case["question"],
            ground_truth=get_ground_truth(case),
            text_agent_output=result["text_output"],
            vision_agent_output=result["vision_output"],
            meta_output=result["meta_output"],
            brier_score=case.get("brier_score", 0.0),
        )

    # ── RAG-specific summary ──────────────────────────────────────────
    n_run = evaluator.total
    if n_run > 0:
        avg_passages = total_passages_retrieved / n_run
        wandb.summary["rag/avg_passages_per_case"] = avg_passages
        wandb.summary["rag/empty_retrievals"] = empty_retrievals
        wandb.summary["rag/top_k"] = top_k

        print(f"\n  RAG Stats:")
        print(f"    Avg passages per case: {avg_passages:.1f}")
        print(f"    Empty retrievals: {empty_retrievals}/{n_run}")

    evaluator.finish()


if __name__ == "__main__":
    asyncio.run(main())