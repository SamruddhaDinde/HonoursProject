"""
Mode 3: Structured JSON Communication on NEJM Image Challenge.

Single-round pipeline where specialists output typed JSON with confidence
scores and named findings. The meta agent receives structured data rather
than free-text reasoning, enabling principled arbitration based on
confidence levels and explicit evidence.

Key difference from Mode 2: same underlying information, different format.
If Mode 3 outperforms Mode 2a despite receiving the same information in a
different structure, that demonstrates communication FORMAT matters
independently of communication CONTENT.

Usage:
    # ThoughtComm-aligned test split (339 cases)
    python main_structured.py --split test

    # Full dataset (689 cases) — for 3-way comparison
    python main_structured.py

    # Quick smoke test
    python main_structured.py --n 5
"""

import os
import json
import asyncio
import argparse
from dotenv import load_dotenv
from agents import Runner

from diff_agents.textAgent import text_agent
from diff_agents.visionAgent import vision_agent
from diff_agents.metaAgent import meta_agent
from data.loader import (
    load_nejm,
    format_text_question_structured,
    format_vision_question_structured,
    format_meta_question_structured,
    get_ground_truth,
    image_to_base64,
    _format_options,
)
from evaluation.evaluator import extract_answer, is_correct
from evaluation.json_parser import parse_specialist_json, build_retry_prompt

import wandb

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

COMMUNICATION_MODE = "structured_json"
SEED = 42
TRAIN_SPLIT = 350
MAX_RETRIES = 1  # one retry on JSON parse failure


# ── Pipeline ─────────────────────────────────────────────────────────────

async def get_structured_output(agent, prompt, agent_name: str, is_vision: bool = False,
                                 image_b64: str = None) -> tuple[dict, str, str, int]:
    """Run an agent and parse its JSON output, with one retry on failure.

    Args:
        agent: the Agent instance
        prompt: formatted prompt string
        agent_name: for logging ("text" or "vision")
        is_vision: if True, send multimodal input with image
        image_b64: base64 image string (required if is_vision)

    Returns:
        (parsed_json, raw_output, parse_status, attempts)
        where parse_status is "clean", "stripped", "partial", or "failed"
        and attempts is 1 or 2
    """
    # Build input
    if is_vision and image_b64:
        agent_input = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        agent_input = prompt

    # First attempt
    result = await Runner.run(agent, agent_input)
    
    raw_output = result.final_output
    parsed, status = parse_specialist_json(raw_output)

    if status in ("clean", "stripped"):
        return parsed, raw_output, status, 1

    # Retry once with corrective prompt
    if MAX_RETRIES > 0:
        retry_prompt = build_retry_prompt(prompt, raw_output)
        # Retry is always text-only (no image re-send on retry)
        result_retry = await Runner.run(agent, retry_prompt)
        raw_retry = result_retry.final_output
        parsed_retry, status_retry = parse_specialist_json(raw_retry)

        # Use retry result if it's better than original
        if status_retry in ("clean", "stripped"):
            return parsed_retry, raw_retry, f"retry_{status_retry}", 2
        elif status_retry == "partial" and status == "failed":
            return parsed_retry, raw_retry, "retry_partial", 2

    # Return best effort from first attempt
    return parsed, raw_output, status, (2 if MAX_RETRIES > 0 else 1)


async def run_structured_pipeline(example: dict) -> dict:
    """
    Mode 3: Structured JSON communication.

    1. Text agent produces structured JSON from clinical text
    2. Vision agent produces structured JSON from image
    3. Meta agent receives both structured assessments and arbitrates
    """

    base64_image = image_to_base64(example["image"])

    # ── Text agent (structured JSON output) ───────────────────────────
    text_prompt = format_text_question_structured(example)
    text_json, text_raw, text_status, text_attempts = await get_structured_output(
        text_agent, text_prompt, "text"
    )

    # ── Vision agent (structured JSON output) ─────────────────────────
    vision_prompt = format_vision_question_structured(example)
    vision_json, vision_raw, vision_status, vision_attempts = await get_structured_output(
        vision_agent, vision_prompt, "vision",
        is_vision=True, image_b64=base64_image
    )

    # ── Meta agent (receives structured data, constrained options) ────
    meta_input = format_meta_question_structured(example, text_json, vision_json)
    result_meta = await Runner.run(meta_agent, meta_input)
    meta_output = result_meta.final_output

    return {
        "text_json": text_json,
        "text_raw": text_raw,
        "text_status": text_status,
        "text_attempts": text_attempts,
        "vision_json": vision_json,
        "vision_raw": vision_raw,
        "vision_status": vision_status,
        "vision_attempts": vision_attempts,
        "meta_output": meta_output,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mode 3: Structured JSON on NEJM")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
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
            "max_retries": MAX_RETRIES,
            "schema_fields": ["answer", "confidence", "key_findings",
                              "supporting_evidence", "alternative_considered",
                              "why_not_alternative"],
        }
    )

    # ── Table schema ──────────────────────────────────────────────────

    table = wandb.Table(columns=[
        "image_id",
        "question",
        "ground_truth",
        "brier_score",

        # Text agent structured output
        "text_answer",
        "text_correct",
        "text_confidence",
        "text_key_findings",
        "text_alternative",
        "text_parse_status",
        "text_attempts",

        # Vision agent structured output
        "vision_answer",
        "vision_correct",
        "vision_confidence",
        "vision_key_findings",
        "vision_alternative",
        "vision_parse_status",
        "vision_attempts",

        # Agreement and confidence comparison
        "agents_agree",
        "confidence_delta",  # text_conf - vision_conf
        "higher_confidence_correct",

        # Meta agent (constrained)
        "meta_answer",
        "meta_correct",
        "meta_followed_higher_confidence",

        # Raw outputs for qualitative analysis
        "text_raw_output",
        "vision_raw_output",
        "meta_output",

        # Running metrics
        "running_meta_accuracy",
        "running_text_accuracy",
        "running_vision_accuracy",
    ])

    # ── Counters ──────────────────────────────────────────────────────

    total = 0
    meta_correct_count = 0
    text_correct_count = 0
    vision_correct_count = 0
    agreement_count = 0

    # Parse tracking
    text_parse_counts = {"clean": 0, "stripped": 0, "partial": 0, "failed": 0,
                         "retry_clean": 0, "retry_stripped": 0, "retry_partial": 0}
    vision_parse_counts = {"clean": 0, "stripped": 0, "partial": 0, "failed": 0,
                           "retry_clean": 0, "retry_stripped": 0, "retry_partial": 0}
    total_retries = 0

    # Confidence tracking
    confidence_sum_text = 0.0
    confidence_sum_vision = 0.0
    higher_conf_correct_count = 0
    higher_conf_total = 0
    meta_followed_higher_count = 0

    # Disagreement tracking
    disagree_total = 0
    disagree_meta_correct = 0
    agree_total = 0
    agree_meta_correct = 0

    # ── Run evaluation ────────────────────────────────────────────────

    for i, case in enumerate(dataset):
        print(f"\n[{i+1}/{n_cases}] Case {case['image_id']:04d}: "
              f"{case['question'][:80]}...")

        #result = await run_structured_pipeline(case)
        try:
            result = await run_structured_pipeline(case)
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            continue
        ground_truth = get_ground_truth(case)

        # Extract answers
        t_json = result["text_json"]
        v_json = result["vision_json"]
        t_ans = t_json.get("answer", "?")
        v_ans = v_json.get("answer", "?")
        t_conf = t_json.get("confidence", 0.5)
        v_conf = v_json.get("confidence", 0.5)
        meta_ans = extract_answer(result["meta_output"])

        # Score
        t_correct = is_correct(t_ans, ground_truth)
        v_correct = is_correct(v_ans, ground_truth)
        meta_ok = is_correct(meta_ans, ground_truth)
        agents_agree = (t_ans == v_ans)

        # Confidence analysis
        conf_delta = t_conf - v_conf
        higher_conf_agent = "text" if t_conf >= v_conf else "vision"
        higher_conf_ans = t_ans if higher_conf_agent == "text" else v_ans
        higher_conf_is_correct = is_correct(higher_conf_ans, ground_truth)
        meta_followed_higher = (meta_ans == higher_conf_ans)

        # Tally
        total += 1
        if t_correct: text_correct_count += 1
        if v_correct: vision_correct_count += 1
        if meta_ok: meta_correct_count += 1

        confidence_sum_text += t_conf
        confidence_sum_vision += v_conf

        if not agents_agree:
            higher_conf_total += 1
            if higher_conf_is_correct: higher_conf_correct_count += 1
            if meta_followed_higher: meta_followed_higher_count += 1

        if agents_agree:
            agreement_count += 1
            agree_total += 1
            if meta_ok: agree_meta_correct += 1
        else:
            disagree_total += 1
            if meta_ok: disagree_meta_correct += 1

        # Parse tracking
        t_status = result["text_status"]
        v_status = result["vision_status"]
        text_parse_counts[t_status] = text_parse_counts.get(t_status, 0) + 1
        vision_parse_counts[v_status] = vision_parse_counts.get(v_status, 0) + 1
        if result["text_attempts"] > 1: total_retries += 1
        if result["vision_attempts"] > 1: total_retries += 1

        # Running metrics
        run_meta = meta_correct_count / total
        run_text = text_correct_count / total
        run_vision = vision_correct_count / total

        # Table row
        table.add_data(
            case["image_id"],
            case["question"][:150] + "..." if len(case["question"]) > 150
                else case["question"],
            ground_truth,
            round(case.get("brier_score", 0.0), 4),

            # Text
            t_ans,
            "correct" if t_correct else "wrong",
            round(t_conf, 3),
            "; ".join(t_json.get("key_findings", [])),
            str(t_json.get("alternative_considered", "N/A")),
            t_status,
            result["text_attempts"],

            # Vision
            v_ans,
            "correct" if v_correct else "wrong",
            round(v_conf, 3),
            "; ".join(v_json.get("key_findings", [])),
            str(v_json.get("alternative_considered", "N/A")),
            v_status,
            result["vision_attempts"],

            # Agreement and confidence
            "yes" if agents_agree else "no",
            round(conf_delta, 3),
            ("yes" if higher_conf_is_correct else "no") if not agents_agree else "N/A",

            # Meta
            meta_ans,
            "correct" if meta_ok else "wrong",
            ("yes" if meta_followed_higher else "no") if not agents_agree else "N/A",

            # Raw outputs
            result["text_raw"],
            result["vision_raw"],
            result["meta_output"],

            # Running
            round(run_meta, 4),
            round(run_text, 4),
            round(run_vision, 4),
        )

        # Live charts
        wandb.log({
            "running/meta_accuracy": run_meta,
            "running/text_accuracy": run_text,
            "running/vision_accuracy": run_vision,
            "running/agreement_rate": agreement_count / total,
            "running/avg_text_confidence": confidence_sum_text / total,
            "running/avg_vision_confidence": confidence_sum_vision / total,
        })

        # Console
        print(f"  Text: {t_ans}(conf={t_conf:.2f},{'ok' if t_correct else 'x'},{t_status}) | "
              f"Vision: {v_ans}(conf={v_conf:.2f},{'ok' if v_correct else 'x'},{v_status}) | "
              f"{'AGREE' if agents_agree else 'DISAGREE'} | "
              f"Meta: {meta_ans}({'ok' if meta_ok else 'x'}) | GT: {ground_truth}")

    # ── Summary ───────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Mode 3 (Structured JSON) Results — {total} cases")
    print(f"{'='*60}")

    print(f"\n  Accuracy:")
    print(f"    Text agent:   {text_correct_count}/{total} ({text_correct_count/total:.1%})")
    print(f"    Vision agent: {vision_correct_count}/{total} ({vision_correct_count/total:.1%})")
    print(f"    Meta agent:   {meta_correct_count}/{total} ({meta_correct_count/total:.1%})")

    print(f"\n  Agreement: {agreement_count}/{total} ({agreement_count/total:.1%})")

    print(f"\n  Confidence:")
    print(f"    Avg text confidence:   {confidence_sum_text/total:.3f}")
    print(f"    Avg vision confidence: {confidence_sum_vision/total:.3f}")
    if higher_conf_total > 0:
        print(f"    Higher-confidence agent correct (on disagreements): "
              f"{higher_conf_correct_count}/{higher_conf_total} "
              f"({higher_conf_correct_count/higher_conf_total:.1%})")
        print(f"    Meta followed higher confidence: "
              f"{meta_followed_higher_count}/{higher_conf_total} "
              f"({meta_followed_higher_count/higher_conf_total:.1%})")

    print(f"\n  Meta Accuracy by Agreement:")
    if agree_total > 0:
        print(f"    When agents agree:    {agree_meta_correct}/{agree_total} "
              f"({agree_meta_correct/agree_total:.1%})")
    if disagree_total > 0:
        print(f"    When agents disagree: {disagree_meta_correct}/{disagree_total} "
              f"({disagree_meta_correct/disagree_total:.1%})")

    print(f"\n  JSON Parse Quality:")
    print(f"    Text agent:  {text_parse_counts}")
    print(f"    Vision agent: {vision_parse_counts}")
    print(f"    Total retries needed: {total_retries}")

    # ── W&B summary ──────────────────────────────────────────────────

    summary = {
        "results_table": table,
        "final/meta_accuracy": meta_correct_count / total,
        "final/text_accuracy": text_correct_count / total,
        "final/vision_accuracy": vision_correct_count / total,
        "final/agreement_rate": agreement_count / total,
        "final/avg_text_confidence": confidence_sum_text / total,
        "final/avg_vision_confidence": confidence_sum_vision / total,
        "final/total_cases": total,
        "final/total_retries": total_retries,
        "final/agree_total": agree_total,
        "final/disagree_total": disagree_total,
    }

    if agree_total > 0:
        summary["final/meta_accuracy_when_agree"] = agree_meta_correct / agree_total
    if disagree_total > 0:
        summary["final/meta_accuracy_when_disagree"] = disagree_meta_correct / disagree_total
    if higher_conf_total > 0:
        summary["final/higher_confidence_correct_rate"] = higher_conf_correct_count / higher_conf_total
        summary["final/meta_followed_higher_confidence_rate"] = meta_followed_higher_count / higher_conf_total

    # Parse quality
    for status, count in text_parse_counts.items():
        summary[f"final/text_parse_{status}"] = count
    for status, count in vision_parse_counts.items():
        summary[f"final/vision_parse_{status}"] = count

    wandb.log(summary)
    for key, val in summary.items():
        if key != "results_table" and isinstance(val, (int, float)):
            wandb.summary[key] = val

    print(f"\n{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())