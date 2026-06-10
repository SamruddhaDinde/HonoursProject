"""
Mode 3b: Structured JSON Debate on NEJM Image Challenge.

Two-round pipeline where agents communicate in structured JSON:
  Round 1: Both agents produce typed JSON independently (same as Mode 3)
  Exchange: Each agent sees the other's structured assessment and revises (in JSON)
  Round 2: Meta agent receives both revised structured assessments

This completes the 2x2 factorial design:
  |              | Single-round | Multi-round     |
  |--------------|------------- |-----------------|
  | Free text    | Mode 1 (CoT) | Mode 2 (debate) |
  | Structured   | Mode 3 (JSON)| Mode 3b (this)  |

The key research question: does structured format prevent the persuasion
bias observed in free-text debate (Mode 2)? If the text agent does NOT
degrade during structured debate, that isolates the format as the cause
of persuasion bias, not the revision step itself.

Usage:
    python main_structured_debate.py --split test    # 339 cases
    python main_structured_debate.py                  # all 689 cases
    python main_structured_debate.py --n 5           # smoke test
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
    split_context_and_question,
    _format_options,
)
from evaluation.evaluator import extract_answer, is_correct
from evaluation.json_parser import parse_specialist_json, build_retry_prompt

import wandb

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

COMMUNICATION_MODE = "structured_json_debate"
SEED = 42
TRAIN_SPLIT = 350
MAX_RETRIES = 1


# ── Structured revision prompts ──────────────────────────────────────────

def _format_structured_assessment(agent_name: str, parsed_json: dict) -> str:
    """Format a parsed JSON assessment into readable structured text."""
    return f"""── {agent_name} ──
Answer: {parsed_json.get('answer', '?')}
Confidence: {parsed_json.get('confidence', 'N/A')}
Key findings: {', '.join(parsed_json.get('key_findings', [])) or 'None provided'}
Supporting evidence: {parsed_json.get('supporting_evidence', 'None provided')}
Alternative considered: {parsed_json.get('alternative_considered', 'N/A')}
Why rejected: {parsed_json.get('why_not_alternative', 'N/A')}"""


def format_text_revision_structured(example: dict, text_r1_json: dict, vision_r1_json: dict) -> str:
    """R2 prompt for text agent: sees vision's structured R1 assessment, revises in JSON."""
    _, diagnostic_question = split_context_and_question(example["question"])

    return f"""You previously assessed a clinical case and produced a structured diagnosis.
Now you are shown the radiology vision specialist's independent structured assessment
of the same case. They examined the medical image for this query:
"{diagnostic_question}"

Your original assessment:
{_format_structured_assessment("Your Assessment", text_r1_json)}

Vision Specialist's assessment:
{_format_structured_assessment("Vision Specialist", vision_r1_json)}

Consider whether the vision specialist's findings, confidence level, and reasoning
change your diagnosis. You may keep your original answer or revise it.

Respond with ONLY a valid JSON object. No other text, no markdown backticks.

{{
  "answer": "<single letter A-E>",
  "confidence": <number between 0.0 and 1.0>,
  "key_findings": ["<finding 1>", "<finding 2>"],
  "supporting_evidence": "<one sentence explaining your choice>",
  "alternative_considered": "<single letter A-E of your second choice>",
  "why_not_alternative": "<one sentence why you rejected it>"
}}"""


def format_vision_revision_structured(example: dict, vision_r1_json: dict, text_r1_json: dict) -> str:
    """R2 prompt for vision agent: sees text's structured R1 assessment, revises in JSON."""
    return f"""You previously examined a medical image and produced a structured diagnosis.
Now you are shown the clinical text specialist's independent structured assessment
of the same case. They had access to the full patient history.

Your original assessment:
{_format_structured_assessment("Your Assessment", vision_r1_json)}

Clinical Text Specialist's assessment:
{_format_structured_assessment("Text Specialist", text_r1_json)}

Consider whether the text specialist's clinical findings, confidence level, and reasoning
change your diagnosis. You may keep your original answer or revise it.

Respond with ONLY a valid JSON object. No other text, no markdown backticks.

{{
  "answer": "<single letter A-E>",
  "confidence": <number between 0.0 and 1.0>,
  "key_findings": ["<finding 1>", "<finding 2>"],
  "supporting_evidence": "<one sentence explaining your choice>",
  "alternative_considered": "<single letter A-E of your second choice>",
  "why_not_alternative": "<one sentence why you rejected it>"
}}"""


def format_meta_structured_debate(example: dict, text_r2_json: dict, vision_r2_json: dict) -> str:
    """Meta agent for structured debate: sees revised structured assessments, constrained options."""
    text_ans = text_r2_json.get("answer", "?")
    vision_ans = vision_r2_json.get("answer", "?")
    proposed = set()
    if text_ans in example["options"]:
        proposed.add(text_ans)
    if vision_ans in example["options"]:
        proposed.add(vision_ans)

    if not proposed:
        proposed_options = _format_options(example["options"])
        constraint_note = "Neither specialist produced a clear answer. Choose from all available options."
    else:
        proposed_options = "\n".join(
            f"  {k}) {example['options'][k]}" for k in sorted(proposed)
        )
        constraint_note = "You must choose one of the options proposed by the specialists."

    return f"""You are a senior consultant. Two specialists have assessed a clinical case
through two rounds — first independently, then after reviewing each other's structured
assessments including confidence scores and specific findings.
You are seeing their final revised structured assessments.

You do not have access to the original case details. {constraint_note}

The specialists proposed these options:
{proposed_options}

{_format_structured_assessment("Clinical Text Specialist (Revised)", text_r2_json)}

{_format_structured_assessment("Radiology Vision Specialist (Revised)", vision_r2_json)}

Based on the specialists' revised structured assessments, confidence levels, and findings,
choose the most likely diagnosis.
Respond in this exact format:
ANSWER: <single letter from the proposed options above>
REASONING: <why you chose this, referencing confidence and findings, 2-3 sentences>"""


# ── Pipeline helpers ─────────────────────────────────────────────────────

async def get_structured_output(agent, prompt, agent_name: str, is_vision: bool = False,
                                 image_b64: str = None) -> tuple[dict, str, str, int]:
    """Run an agent and parse JSON output with one retry on failure."""
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

    result = await Runner.run(agent, agent_input)
    raw_output = result.final_output
    parsed, status = parse_specialist_json(raw_output)

    if status in ("clean", "stripped"):
        return parsed, raw_output, status, 1

    if MAX_RETRIES > 0:
        retry_prompt = build_retry_prompt(prompt, raw_output)
        result_retry = await Runner.run(agent, retry_prompt)
        raw_retry = result_retry.final_output
        parsed_retry, status_retry = parse_specialist_json(raw_retry)

        if status_retry in ("clean", "stripped"):
            return parsed_retry, raw_retry, f"retry_{status_retry}", 2
        elif status_retry == "partial" and status == "failed":
            return parsed_retry, raw_retry, "retry_partial", 2

    return parsed, raw_output, status, (2 if MAX_RETRIES > 0 else 1)


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


# ── Main pipeline ────────────────────────────────────────────────────────

async def run_structured_debate_pipeline(example: dict) -> dict:
    """
    Two-round structured JSON debate pipeline.

    R1: Both agents produce JSON independently
    Exchange: Each agent sees the other's structured assessment, revises in JSON
    R2: Meta agent synthesises revised structured outputs
    """
    base64_image = image_to_base64(example["image"])

    # ── Round 1: Independent structured answers ───────────────────────

    text_r1_prompt = format_text_question_structured(example)
    text_r1_json, text_r1_raw, text_r1_status, text_r1_attempts = await get_structured_output(
        text_agent, text_r1_prompt, "text_r1"
    )

    vision_r1_prompt = format_vision_question_structured(example)
    vision_r1_json, vision_r1_raw, vision_r1_status, vision_r1_attempts = await get_structured_output(
        vision_agent, vision_r1_prompt, "vision_r1",
        is_vision=True, image_b64=base64_image
    )

    # ── Exchange: Each agent sees other's structured R1, revises ──────

    text_r2_prompt = format_text_revision_structured(example, text_r1_json, vision_r1_json)
    text_r2_json, text_r2_raw, text_r2_status, text_r2_attempts = await get_structured_output(
        text_agent, text_r2_prompt, "text_r2"
    )

    vision_r2_prompt = format_vision_revision_structured(example, vision_r1_json, text_r1_json)
    vision_r2_json, vision_r2_raw, vision_r2_status, vision_r2_attempts = await get_structured_output(
        vision_agent, vision_r2_prompt, "vision_r2"
    )

    # ── Meta synthesis: revised structured outputs, constrained ───────

    meta_input = format_meta_structured_debate(example, text_r2_json, vision_r2_json)
    result_meta = await Runner.run(meta_agent, meta_input)
    meta_output = result_meta.final_output

    return {
        "text_r1_json": text_r1_json,
        "text_r1_raw": text_r1_raw,
        "text_r1_status": text_r1_status,
        "text_r2_json": text_r2_json,
        "text_r2_raw": text_r2_raw,
        "text_r2_status": text_r2_status,
        "vision_r1_json": vision_r1_json,
        "vision_r1_raw": vision_r1_raw,
        "vision_r1_status": vision_r1_status,
        "vision_r2_json": vision_r2_json,
        "vision_r2_raw": vision_r2_raw,
        "vision_r2_status": vision_r2_status,
        "meta_output": meta_output,
        "total_attempts": (text_r1_attempts + text_r2_attempts +
                          vision_r1_attempts + vision_r2_attempts),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mode 3b: Structured JSON debate on NEJM")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


async def main():
    args = parse_args()

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
            "output_format": "structured_json",
            "method": "structured_debate",
        }
    )

    # ── Table schema ──────────────────────────────────────────────────

    table = wandb.Table(columns=[
        "image_id", "question", "ground_truth", "brier_score",

        # R1 structured
        "text_r1_answer", "text_r1_correct", "text_r1_confidence",
        "vision_r1_answer", "vision_r1_correct", "vision_r1_confidence",
        "r1_agree",

        # R2 structured
        "text_r2_answer", "text_r2_correct", "text_r2_confidence",
        "vision_r2_answer", "vision_r2_correct", "vision_r2_confidence",
        "r2_agree",

        # Change tracking
        "text_changed", "text_change_direction",
        "vision_changed", "vision_change_direction",

        # Confidence change
        "text_conf_delta", "vision_conf_delta",

        # Meta
        "meta_answer", "meta_correct",

        # Parse quality
        "text_r1_parse", "text_r2_parse",
        "vision_r1_parse", "vision_r2_parse",

        # Raw outputs
        "text_r1_raw", "vision_r1_raw",
        "text_r2_raw", "vision_r2_raw",
        "meta_output",

        # Running
        "running_meta_accuracy", "running_text_r1_accuracy",
        "running_text_r2_accuracy", "running_vision_r1_accuracy",
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
    text_improved = 0
    text_degraded = 0
    vision_improved = 0
    vision_degraded = 0
    r1_agree_count = 0
    r2_agree_count = 0
    total_retries = 0

    # Confidence tracking
    conf_sum_t_r1 = 0.0
    conf_sum_t_r2 = 0.0
    conf_sum_v_r1 = 0.0
    conf_sum_v_r2 = 0.0

    # Disagreement-conditioned
    r2_agree_total = 0
    r2_agree_meta_correct = 0
    r2_disagree_total = 0
    r2_disagree_meta_correct = 0

    # ── Run ───────────────────────────────────────────────────────────

    for i, case in enumerate(dataset):
        print(f"\n[{i+1}/{n_cases}] Case {case['image_id']:04d}: "
              f"{case['question'][:80]}...")

        try:
            result = await run_structured_debate_pipeline(case)
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            continue

        ground_truth = get_ground_truth(case)

        # Extract answers and confidence
        t_r1_json = result["text_r1_json"]
        t_r2_json = result["text_r2_json"]
        v_r1_json = result["vision_r1_json"]
        v_r2_json = result["vision_r2_json"]

        t_r1 = t_r1_json.get("answer", "?")
        t_r2 = t_r2_json.get("answer", "?")
        v_r1 = v_r1_json.get("answer", "?")
        v_r2 = v_r2_json.get("answer", "?")
        t_r1_conf = t_r1_json.get("confidence", 0.5)
        t_r2_conf = t_r2_json.get("confidence", 0.5)
        v_r1_conf = v_r1_json.get("confidence", 0.5)
        v_r2_conf = v_r2_json.get("confidence", 0.5)
        meta_ans = extract_answer(result["meta_output"])

        # Score
        t_r1_ok = is_correct(t_r1, ground_truth)
        t_r2_ok = is_correct(t_r2, ground_truth)
        v_r1_ok = is_correct(v_r1, ground_truth)
        v_r2_ok = is_correct(v_r2, ground_truth)
        meta_ok = is_correct(meta_ans, ground_truth)

        r1_agree = (t_r1 == v_r1)
        r2_agree = (t_r2 == v_r2)
        t_changed = (t_r1 != t_r2)
        v_changed = (v_r1 != v_r2)
        t_dir = change_direction(t_r1, t_r2, ground_truth)
        v_dir = change_direction(v_r1, v_r2, ground_truth)

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
        if t_dir == "wrong_to_right": text_improved += 1
        if t_dir == "right_to_wrong": text_degraded += 1
        if v_dir == "wrong_to_right": vision_improved += 1
        if v_dir == "right_to_wrong": vision_degraded += 1
        total_retries += result["total_attempts"] - 4  # 4 calls is baseline

        conf_sum_t_r1 += t_r1_conf
        conf_sum_t_r2 += t_r2_conf
        conf_sum_v_r1 += v_r1_conf
        conf_sum_v_r2 += v_r2_conf

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

            t_r1, "correct" if t_r1_ok else "wrong", round(t_r1_conf, 3),
            v_r1, "correct" if v_r1_ok else "wrong", round(v_r1_conf, 3),
            "yes" if r1_agree else "no",

            t_r2, "correct" if t_r2_ok else "wrong", round(t_r2_conf, 3),
            v_r2, "correct" if v_r2_ok else "wrong", round(v_r2_conf, 3),
            "yes" if r2_agree else "no",

            "yes" if t_changed else "no", t_dir,
            "yes" if v_changed else "no", v_dir,

            round(t_r2_conf - t_r1_conf, 3),
            round(v_r2_conf - v_r1_conf, 3),

            meta_ans, "correct" if meta_ok else "wrong",

            result["text_r1_status"], result["text_r2_status"],
            result["vision_r1_status"], result["vision_r2_status"],

            result["text_r1_raw"], result["vision_r1_raw"],
            result["text_r2_raw"], result["vision_r2_raw"],
            result["meta_output"],

            round(run_meta, 4), round(run_t_r1, 4), round(run_t_r2, 4),
            round(run_v_r1, 4), round(run_v_r2, 4),
        )

        # Live charts
        wandb.log({
            "running/meta_accuracy": run_meta,
            "running/text_r1_accuracy": run_t_r1,
            "running/text_r2_accuracy": run_t_r2,
            "running/vision_r1_accuracy": run_v_r1,
            "running/vision_r2_accuracy": run_v_r2,
            "running/r2_agreement_rate": r2_agree_count / total,
        })

        # Console
        print(f"  Text:   {t_r1}({t_r1_conf:.2f})→{t_r2}({t_r2_conf:.2f}) [{t_dir}] | "
              f"Vision: {v_r1}({v_r1_conf:.2f})→{v_r2}({v_r2_conf:.2f}) [{v_dir}] | "
              f"Meta: {meta_ans}({'ok' if meta_ok else 'x'}) | GT: {ground_truth}")

    # ── Summary ───────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Mode 3b (Structured JSON Debate) Results — {total} cases")
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
    print(f"      Improved: {text_improved} | Degraded: {text_degraded} | "
          f"Net: {text_improved - text_degraded:+d}")
    print(f"    Vision changed: {vision_changed_count}/{total} ({vision_changed_count/total:.1%})")
    print(f"      Improved: {vision_improved} | Degraded: {vision_degraded} | "
          f"Net: {vision_improved - vision_degraded:+d}")

    print(f"\n  Confidence (avg):")
    print(f"    Text  R1: {conf_sum_t_r1/total:.3f} → R2: {conf_sum_t_r2/total:.3f}")
    print(f"    Vision R1: {conf_sum_v_r1/total:.3f} → R2: {conf_sum_v_r2/total:.3f}")

    print(f"\n  Meta Accuracy by R2 Agreement:")
    if r2_agree_total > 0:
        print(f"    When R2 agree:    {r2_agree_meta_correct}/{r2_agree_total} "
              f"({r2_agree_meta_correct/r2_agree_total:.1%})")
    if r2_disagree_total > 0:
        print(f"    When R2 disagree: {r2_disagree_meta_correct}/{r2_disagree_total} "
              f"({r2_disagree_meta_correct/r2_disagree_total:.1%})")

    print(f"\n  Parse quality: {total_retries} retries needed")

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
        "final/text_improved": text_improved,
        "final/text_degraded": text_degraded,
        "final/text_net": text_improved - text_degraded,
        "final/vision_improved": vision_improved,
        "final/vision_degraded": vision_degraded,
        "final/vision_net": vision_improved - vision_degraded,
        "final/avg_text_r1_confidence": conf_sum_t_r1 / total,
        "final/avg_text_r2_confidence": conf_sum_t_r2 / total,
        "final/avg_vision_r1_confidence": conf_sum_v_r1 / total,
        "final/avg_vision_r2_confidence": conf_sum_v_r2 / total,
        "final/total_retries": total_retries,
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