"""
Mode 2b: Exchange-of-Thought debate pipeline on NEJM Image Challenge.

Two-round pipeline:
  Round 1: Text and vision agents answer independently
  Exchange: Each agent sees the other's R1 reasoning and revises
  Round 2: Meta agent synthesises the revised (R2) outputs

This mirrors ThoughtComm's two-round structure but uses natural language
reasoning traces instead of latent hidden-state representations.

Methodological precedent: Exchange-of-Thought (Yin et al., EMNLP 2023)

Usage:
    # ThoughtComm-aligned test split (339 cases) — for 4-way comparison
    python main_debate.py --split test

    # Full dataset (689 cases) — for 3-way text-level comparison
    python main_debate.py

    # Quick smoke test
    python main_debate.py --n 5
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
    format_text_revision,
    format_vision_revision,
    format_meta_question_debate,
    get_ground_truth,
    image_to_base64,
)
from evaluation.evaluator import extract_answer, is_correct

import wandb

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

COMMUNICATION_MODE = "cot_debate"
SEED = 42
TRAIN_SPLIT = 350


# ── Pipeline ─────────────────────────────────────────────────────────────

async def run_debate_pipeline(example: dict) -> dict:
    """
    Two-round Exchange-of-Thought pipeline.

    Round 1: Both agents answer independently (identical to output-only)
    Exchange: Each agent receives the other's R1 output and revises
    Round 2: Meta agent synthesises the revised outputs

    Returns a dict with all intermediate outputs for comprehensive logging.
    """

    # ── Round 1: Independent answers ──────────────────────────────────

    # Text agent R1
    text_question = format_text_question(example)
    result_text_r1 = await Runner.run(text_agent, text_question)
    text_r1 = result_text_r1.final_output

    # Vision agent R1
    base64_image = image_to_base64(example["image"])
    vision_question = format_vision_question(example)

    vision_input_r1 = [
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

    result_vision_r1 = await Runner.run(vision_agent, vision_input_r1)
    vision_r1 = result_vision_r1.final_output

    # ── Exchange: Each agent sees the other's R1 reasoning ────────────

    # Text agent R2: sees vision's R1 reasoning, revises
    text_revision_prompt = format_text_revision(example, text_r1, vision_r1)
    result_text_r2 = await Runner.run(text_agent, text_revision_prompt)
    text_r2 = result_text_r2.final_output

    # Vision agent R2: sees text's R1 reasoning, revises
    # Note: vision agent gets revision as TEXT-ONLY (no image re-sent).
    # The revision is about integrating the text specialist's clinical
    # reasoning, not re-examining the image.
    vision_revision_prompt = format_vision_revision(example, vision_r1, text_r1)
    result_vision_r2 = await Runner.run(vision_agent, vision_revision_prompt)
    vision_r2 = result_vision_r2.final_output

    # ── Meta synthesis: sees revised (R2) outputs only ────────────────

    meta_input = format_meta_question_debate(example, text_r2, vision_r2)
    result_meta = await Runner.run(meta_agent, meta_input)
    meta_output = result_meta.final_output

    return {
        "text_r1": text_r1,
        "vision_r1": vision_r1,
        "text_r2": text_r2,
        "vision_r2": vision_r2,
        "meta_output": meta_output,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mode 2b: CoT debate on NEJM")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def change_direction(r1_ans, r2_ans, gt):
    """Classify how an agent's answer changed between rounds."""
    if r1_ans == r2_ans:
        return "no_change"
    r1_right = is_correct(r1_ans, gt)
    r2_right = is_correct(r2_ans, gt)
    if not r1_right and r2_right:
        return "wrong_to_right"
    elif r1_right and not r2_right:
        return "right_to_wrong"
    else:
        return "wrong_to_wrong"


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
    print(f"Running {COMMUNICATION_MODE} on {n_cases} NEJM cases "
          f"(split={args.split}, seed={args.seed})")

    # ── W&B init ──────────────────────────────────────────────────────

    run = wandb.init(
        project="medical-multiagent",
        name=f"{COMMUNICATION_MODE}_{split_label}_seed{args.seed}",
        config={
            "communication_mode": COMMUNICATION_MODE,
            "model": "medgemma1.5:4b",
            "meta_model": "medgemma1.5:4b",
            "dataset": "NEJM",
            "split": args.split,
            "n_samples": n_cases,
            "seed": args.seed,
            "train_split": TRAIN_SPLIT,
            "num_rounds": 2,
            "method": "exchange_of_thought",
        }
    )

    # ── Table schema (aligned with ThoughtComm for comparison) ────────

    table = wandb.Table(columns=[
        "image_id",
        "question",
        "ground_truth",
        "brier_score",

        # Round 1 (independent)
        "text_r1_answer",
        "text_r1_correct",
        "vision_r1_answer",
        "vision_r1_correct",
        "r1_agree",

        # Round 2 (after exchange)
        "text_r2_answer",
        "text_r2_correct",
        "vision_r2_answer",
        "vision_r2_correct",
        "r2_agree",

        # Change tracking
        "text_changed",
        "text_change_direction",
        "vision_changed",
        "vision_change_direction",

        # Meta (system output)
        "meta_answer",
        "meta_correct",

        # Full outputs
        "text_r1_output",
        "vision_r1_output",
        "text_r2_output",
        "vision_r2_output",
        "meta_output",

        # Running metrics
        "running_meta_accuracy",
        "running_text_r1_accuracy",
        "running_text_r2_accuracy",
        "running_vision_r1_accuracy",
        "running_vision_r2_accuracy",
    ])

    # ── Counters ──────────────────────────────────────────────────────

    total = 0
    meta_correct_count = 0
    text_r1_correct_count = 0
    text_r2_correct_count = 0
    vision_r1_correct_count = 0
    vision_r2_correct_count = 0
    text_changed_count = 0
    vision_changed_count = 0
    text_improved_count = 0
    text_degraded_count = 0
    vision_improved_count = 0
    vision_degraded_count = 0
    r1_agree_count = 0
    r2_agree_count = 0

    # Disagreement-conditioned
    r2_disagree_total = 0
    r2_disagree_meta_correct = 0
    r2_agree_total = 0
    r2_agree_meta_correct = 0

    # ── Run evaluation ────────────────────────────────────────────────

    for i, case in enumerate(dataset):
        print(f"\n[{i+1}/{n_cases}] Case {case['image_id']:04d}: "
              f"{case['question'][:80]}...")

        result = await run_debate_pipeline(case)
        ground_truth = get_ground_truth(case)

        # Parse all answers
        t_r1 = extract_answer(result["text_r1"])
        v_r1 = extract_answer(result["vision_r1"])
        t_r2 = extract_answer(result["text_r2"])
        v_r2 = extract_answer(result["vision_r2"])
        meta_ans = extract_answer(result["meta_output"])

        # Score
        t_r1_ok = is_correct(t_r1, ground_truth)
        v_r1_ok = is_correct(v_r1, ground_truth)
        t_r2_ok = is_correct(t_r2, ground_truth)
        v_r2_ok = is_correct(v_r2, ground_truth)
        meta_ok = is_correct(meta_ans, ground_truth)

        r1_agree = (t_r1 == v_r1)
        r2_agree = (t_r2 == v_r2)
        t_changed = (t_r1 != t_r2)
        v_changed = (v_r1 != v_r2)

        t_change_dir = change_direction(t_r1, t_r2, ground_truth)
        v_change_dir = change_direction(v_r1, v_r2, ground_truth)

        # Tally
        total += 1
        if meta_ok: meta_correct_count += 1
        if t_r1_ok: text_r1_correct_count += 1
        if t_r2_ok: text_r2_correct_count += 1
        if v_r1_ok: vision_r1_correct_count += 1
        if v_r2_ok: vision_r2_correct_count += 1
        if r1_agree: r1_agree_count += 1
        if r2_agree: r2_agree_count += 1
        if t_changed: text_changed_count += 1
        if v_changed: vision_changed_count += 1
        if t_change_dir == "wrong_to_right": text_improved_count += 1
        if t_change_dir == "right_to_wrong": text_degraded_count += 1
        if v_change_dir == "wrong_to_right": vision_improved_count += 1
        if v_change_dir == "right_to_wrong": vision_degraded_count += 1

        if r2_agree:
            r2_agree_total += 1
            if meta_ok: r2_agree_meta_correct += 1
        else:
            r2_disagree_total += 1
            if meta_ok: r2_disagree_meta_correct += 1

        # Running metrics
        run_meta = meta_correct_count / total
        run_t_r1 = text_r1_correct_count / total
        run_t_r2 = text_r2_correct_count / total
        run_v_r1 = vision_r1_correct_count / total
        run_v_r2 = vision_r2_correct_count / total

        # Table row
        table.add_data(
            case["image_id"],
            case["question"][:150] + "..." if len(case["question"]) > 150 else case["question"],
            ground_truth,
            round(case.get("brier_score", 0.0), 4),

            t_r1, "correct" if t_r1_ok else "wrong",
            v_r1, "correct" if v_r1_ok else "wrong",
            "yes" if r1_agree else "no",

            t_r2, "correct" if t_r2_ok else "wrong",
            v_r2, "correct" if v_r2_ok else "wrong",
            "yes" if r2_agree else "no",

            "yes" if t_changed else "no", t_change_dir,
            "yes" if v_changed else "no", v_change_dir,

            meta_ans, "correct" if meta_ok else "wrong",

            result["text_r1"],
            result["vision_r1"],
            result["text_r2"],
            result["vision_r2"],
            result["meta_output"],

            round(run_meta, 4),
            round(run_t_r1, 4),
            round(run_t_r2, 4),
            round(run_v_r1, 4),
            round(run_v_r2, 4),
        )

        # Live charts
        wandb.log({
            "running/meta_accuracy": run_meta,
            "running/text_r1_accuracy": run_t_r1,
            "running/text_r2_accuracy": run_t_r2,
            "running/vision_r1_accuracy": run_v_r1,
            "running/vision_r2_accuracy": run_v_r2,
            "running/r2_agreement_rate": r2_agree_count / total,
            "running/text_change_rate": text_changed_count / total,
            "running/vision_change_rate": vision_changed_count / total,
        })

        # Console
        print(f"  Text:   {t_r1}→{t_r2} ({t_change_dir}) | "
              f"Vision: {v_r1}→{v_r2} ({v_change_dir}) | "
              f"Meta: {meta_ans}({'ok' if meta_ok else 'x'}) | GT: {ground_truth}")

    # ── Summary ───────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Mode 2b (CoT Debate) Results — {total} cases")
    print(f"{'='*60}")

    print(f"\n  Individual Agent Accuracy:")
    print(f"    Text  R1: {text_r1_correct_count}/{total} ({text_r1_correct_count/total:.1%})")
    print(f"    Text  R2: {text_r2_correct_count}/{total} ({text_r2_correct_count/total:.1%})")
    print(f"    Vision R1: {vision_r1_correct_count}/{total} ({vision_r1_correct_count/total:.1%})")
    print(f"    Vision R2: {vision_r2_correct_count}/{total} ({vision_r2_correct_count/total:.1%})")
    print(f"    Meta:      {meta_correct_count}/{total} ({meta_correct_count/total:.1%})")

    print(f"\n  Agreement:")
    print(f"    R1: {r1_agree_count}/{total} ({r1_agree_count/total:.1%})")
    print(f"    R2: {r2_agree_count}/{total} ({r2_agree_count/total:.1%})")

    print(f"\n  Exchange Impact (R1→R2 changes):")
    print(f"    Text changed:   {text_changed_count}/{total} ({text_changed_count/total:.1%})")
    print(f"      Improved: {text_improved_count} | Degraded: {text_degraded_count} | "
          f"Net: {text_improved_count - text_degraded_count:+d}")
    print(f"    Vision changed: {vision_changed_count}/{total} ({vision_changed_count/total:.1%})")
    print(f"      Improved: {vision_improved_count} | Degraded: {vision_degraded_count} | "
          f"Net: {vision_improved_count - vision_degraded_count:+d}")

    print(f"\n  Meta Accuracy by R2 Agreement:")
    if r2_agree_total > 0:
        print(f"    When R2 agents agree:    {r2_agree_meta_correct}/{r2_agree_total} "
              f"({r2_agree_meta_correct/r2_agree_total:.1%})")
    if r2_disagree_total > 0:
        print(f"    When R2 agents disagree: {r2_disagree_meta_correct}/{r2_disagree_total} "
              f"({r2_disagree_meta_correct/r2_disagree_total:.1%})")

    # ── W&B summary ──────────────────────────────────────────────────

    summary = {
        "results_table": table,
        "final/meta_accuracy": meta_correct_count / total,
        "final/text_r1_accuracy": text_r1_correct_count / total,
        "final/text_r2_accuracy": text_r2_correct_count / total,
        "final/vision_r1_accuracy": vision_r1_correct_count / total,
        "final/vision_r2_accuracy": vision_r2_correct_count / total,
        "final/r1_agreement_rate": r1_agree_count / total,
        "final/r2_agreement_rate": r2_agree_count / total,
        "final/text_change_rate": text_changed_count / total,
        "final/vision_change_rate": vision_changed_count / total,
        "final/text_improved": text_improved_count,
        "final/text_degraded": text_degraded_count,
        "final/text_net_improvement": text_improved_count - text_degraded_count,
        "final/vision_improved": vision_improved_count,
        "final/vision_degraded": vision_degraded_count,
        "final/vision_net_improvement": vision_improved_count - vision_degraded_count,
        "final/total_cases": total,
    }

    if r2_agree_total > 0:
        summary["final/meta_accuracy_when_r2_agree"] = r2_agree_meta_correct / r2_agree_total
    if r2_disagree_total > 0:
        summary["final/meta_accuracy_when_r2_disagree"] = r2_disagree_meta_correct / r2_disagree_total

    wandb.log(summary)
    for key, val in summary.items():
        if key != "results_table" and isinstance(val, (int, float)):
            wandb.summary[key] = val

    print(f"\n{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())