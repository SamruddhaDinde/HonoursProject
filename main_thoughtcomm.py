"""
PHASE 3: ThoughtComm Evaluation on NEJM Test Set

Runs the full ThoughtComm pipeline on the test set and logs results to W&B.

COMPREHENSIVE LOGGING:
  - Every agent answer at every round logged as separate W&B columns
  - Multiple heuristics evaluated simultaneously (no re-run needed)
  - Per-case analysis: who changed, who was right, what would each heuristic pick
  - Aggregate metrics per heuristic for easy comparison

Run:
    python main_thoughtcomm.py

Prerequisites:
    1. Phase 0 completed: artifacts/hidden_states_{train,test}.pt exist
    2. Phase 1 completed: artifacts/encoder.pt, decoder.pt, structure_mask.pt exist
    3. Phase 2 completed: artifacts/adapter.pt, agreement_weights.pt exist
"""

import os
import sys
import torch
import wandb
from dotenv import load_dotenv

load_dotenv()

from data.loader import (
    load_nejm,
    format_text_question,
    format_vision_question,
    get_ground_truth,
)
from evaluation.evaluator import extract_answer, is_correct
from thoughtcomm.inference import ThoughtCommPipeline


#  Configuration 
ARTIFACTS_DIR = os.getenv("THOUGHTCOMM_ARTIFACTS", "artifacts")
TRAIN_SPLIT = int(os.getenv("THOUGHTCOMM_TRAIN_SPLIT", "350"))
NUM_ROUNDS = int(os.getenv("THOUGHTCOMM_NUM_ROUNDS", "2"))
SEED = 42



# HEURISTICS — all evaluated on every case, logged separately


def heuristic_trust_consistent(v_r1, v_r2, t_r1, t_r2):
    """Trust the agent that did NOT change its answer (original heuristic)."""
    if v_r2 == t_r2:
        return v_r2  # consensus
    if v_r1 == v_r2 and t_r1 != t_r2:
        return v_r2  # vision consistent
    elif t_r1 == t_r2 and v_r1 != v_r2:
        return t_r2  # text consistent
    else:
        return t_r2  # default to text


def heuristic_trust_changed(v_r1, v_r2, t_r1, t_r2):
    """Trust the agent that DID change — ThoughtComm influenced it."""
    if v_r2 == t_r2:
        return v_r2  # consensus
    if v_r1 != v_r2 and t_r1 == t_r2:
        return v_r2  # vision changed (received text thoughts)
    elif t_r1 != t_r2 and v_r1 == v_r2:
        return t_r2  # text changed (received vision thoughts)
    else:
        return t_r2  # both changed, default to text


def heuristic_always_text(v_r1, v_r2, t_r1, t_r2):
    """Always trust the text agent's Round 2 answer."""
    return t_r2


def heuristic_always_vision(v_r1, v_r2, t_r1, t_r2):
    """Always trust the vision agent's Round 2 answer."""
    return v_r2


def heuristic_majority_vote(v_r1, v_r2, t_r1, t_r2):
    """Majority vote across all 4 answers (R1 + R2 for both agents)."""
    votes = [v_r1, v_r2, t_r1, t_r2]
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) == 1:
        return winners[0]
    # Tie-break: prefer R2 answers over R1, text over vision
    for preference in [t_r2, v_r2, t_r1, v_r1]:
        if preference in winners:
            return preference
    return t_r2


def heuristic_weighted_r2(v_r1, v_r2, t_r1, t_r2):
    """R2 answers count double (informed by ThoughtComm), then majority vote."""
    votes = [v_r1, v_r2, v_r2, t_r1, t_r2, t_r2]  # R2 counted twice
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) == 1:
        return winners[0]
    for preference in [t_r2, v_r2, t_r1, v_r1]:
        if preference in winners:
            return preference
    return t_r2


HEURISTICS = {
    "trust_consistent": heuristic_trust_consistent,
    "trust_changed": heuristic_trust_changed,
    "always_text": heuristic_always_text,
    "always_vision": heuristic_always_vision,
    "majority_vote": heuristic_majority_vote,
    "weighted_r2": heuristic_weighted_r2,
}



# MAIN


def main():
    print("=" * 60)
    print("PHASE 3: ThoughtComm Inference on NEJM Test Set")
    print("=" * 60)

    #  Load test cases 
    print(f"\nLoading dataset (seed={SEED})...")
    all_cases = load_nejm(n_samples=None, seed=SEED)
    test_cases = all_cases[TRAIN_SPLIT:]
    n_test = len(test_cases)
    print(f"Test set: {n_test} cases (indices {TRAIN_SPLIT}–{len(all_cases)-1})")

    #  Initialize ThoughtComm pipeline 
    pipeline = ThoughtCommPipeline(ARTIFACTS_DIR)

    #  Initialize W&B ─
    run = wandb.init(
        project="medical-multiagent",
        name=f"thoughtcomm_{NUM_ROUNDS}rounds_{n_test}cases_comprehensive",
        config={
            "communication_mode": "thoughtcomm",
            "num_rounds": NUM_ROUNDS,
            "text_model": "medgemma1.5:4b",
            "vision_model": "medgemma1.5:4b",
            "dataset": "NEJM",
            "n_samples": n_test,
            "train_split": TRAIN_SPLIT,
            "latent_dim": pipeline.latent_dim,
            "n_shared_thoughts": int((pipeline.agreement == 2).sum()),
            "n_private_vision": int(
                (pipeline.vision_mask & ~pipeline.text_mask).sum()),
            "n_private_text": int(
                (pipeline.text_mask & ~pipeline.vision_mask).sum()),
            "w_private": pipeline.w_private.item(),
            "w_shared": pipeline.w_shared.item(),
            "heuristics_evaluated": list(HEURISTICS.keys()),
        },
    )

    #  Comprehensive W&B table 
    columns = [
        # Case info
        "case_num",
        "image_id",
        "question_preview",
        "ground_truth",

        # Round 1: independent answers (before ThoughtComm)
        "vision_r1_answer",
        "text_r1_answer",
        "vision_r1_correct",
        "text_r1_correct",
        "r1_consensus",
        "r1_consensus_correct",

        # Round 2: answers after ThoughtComm
        "vision_r2_answer",
        "text_r2_answer",
        "vision_r2_correct",
        "text_r2_correct",
        "r2_consensus",
        "r2_consensus_correct",

        # Change tracking
        "vision_changed",
        "text_changed",
        "vision_change_direction",
        "text_change_direction",

    ] + [f"heuristic_{name}_answer" for name in HEURISTICS.keys()] \
      + [f"heuristic_{name}_correct" for name in HEURISTICS.keys()] \
      + [
        # Full responses for qualitative analysis
        "vision_r1_response",
        "vision_r2_response",
        "text_r1_response",
        "text_r2_response",
    ]

    results_table = wandb.Table(columns=columns)

    #  Tracking counters 
    heuristic_correct = {name: 0 for name in HEURISTICS}
    heuristic_correct_consensus = {name: 0 for name in HEURISTICS}
    heuristic_correct_disagree = {name: 0 for name in HEURISTICS}

    total = 0
    vision_r1_correct_count = 0
    text_r1_correct_count = 0
    vision_r2_correct_count = 0
    text_r2_correct_count = 0
    r1_consensus_count = 0
    r2_consensus_count = 0
    r1_consensus_correct_count = 0
    r2_consensus_correct_count = 0
    vision_changed_count = 0
    text_changed_count = 0
    vision_improved_count = 0
    vision_degraded_count = 0
    text_improved_count = 0
    text_degraded_count = 0
    r2_consensus_total = 0
    r2_disagree_total = 0

    #  Run evaluation ─
    for i, case in enumerate(test_cases):
        print(f"\n[{i+1}/{n_test}] Case {case['image_id']:04d}")

        result = pipeline.run_case(
            case,
            format_vision_fn=format_vision_question,
            format_text_fn=format_text_question,
            num_rounds=NUM_ROUNDS,
        )

        #  Parse all answers 
        vision_r1 = extract_answer(result["vision_responses"][0])
        text_r1 = extract_answer(result["text_responses"][0])
        vision_r2 = extract_answer(result["vision_responses"][-1])
        text_r2 = extract_answer(result["text_responses"][-1])
        ground_truth = get_ground_truth(case)

        total += 1

        #  Individual agent correctness ─
        v_r1_correct = is_correct(vision_r1, ground_truth)
        t_r1_correct = is_correct(text_r1, ground_truth)
        v_r2_correct = is_correct(vision_r2, ground_truth)
        t_r2_correct = is_correct(text_r2, ground_truth)

        if v_r1_correct: vision_r1_correct_count += 1
        if t_r1_correct: text_r1_correct_count += 1
        if v_r2_correct: vision_r2_correct_count += 1
        if t_r2_correct: text_r2_correct_count += 1

        #  Consensus tracking ─
        r1_consensus = (vision_r1 == text_r1)
        r2_consensus = (vision_r2 == text_r2)

        if r1_consensus:
            r1_consensus_count += 1
            if v_r1_correct:
                r1_consensus_correct_count += 1

        if r2_consensus:
            r2_consensus_count += 1
            r2_consensus_total += 1
            if v_r2_correct:
                r2_consensus_correct_count += 1
        else:
            r2_disagree_total += 1

        #  Change tracking 
        v_changed = (vision_r1 != vision_r2)
        t_changed = (text_r1 != text_r2)

        if v_changed: vision_changed_count += 1
        if t_changed: text_changed_count += 1

        def change_direction(r1_ans, r2_ans, gt):
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

        v_change_dir = change_direction(vision_r1, vision_r2, ground_truth)
        t_change_dir = change_direction(text_r1, text_r2, ground_truth)

        if v_change_dir == "wrong_to_right": vision_improved_count += 1
        if v_change_dir == "right_to_wrong": vision_degraded_count += 1
        if t_change_dir == "wrong_to_right": text_improved_count += 1
        if t_change_dir == "right_to_wrong": text_degraded_count += 1

        #  Apply ALL heuristics ─
        heuristic_answers = {}
        heuristic_correctness = {}
        for name, func in HEURISTICS.items():
            ans = func(vision_r1, vision_r2, text_r1, text_r2)
            correct = is_correct(ans, ground_truth)
            heuristic_answers[name] = ans
            heuristic_correctness[name] = correct
            if correct:
                heuristic_correct[name] += 1
                if r2_consensus:
                    heuristic_correct_consensus[name] += 1
                else:
                    heuristic_correct_disagree[name] += 1

        #  Build table row 
        row = [
            i + 1,
            case["image_id"],
            case["question"][:100] + "..." if len(case["question"]) > 100 else case["question"],
            ground_truth,

            # Round 1
            vision_r1,
            text_r1,
            "correct" if v_r1_correct else "wrong",
            "correct" if t_r1_correct else "wrong",
            "yes" if r1_consensus else "no",
            ("correct" if v_r1_correct else "wrong") if r1_consensus else "N/A",

            # Round 2
            vision_r2,
            text_r2,
            "correct" if v_r2_correct else "wrong",
            "correct" if t_r2_correct else "wrong",
            "yes" if r2_consensus else "no",
            ("correct" if v_r2_correct else "wrong") if r2_consensus else "N/A",

            # Changes
            "yes" if v_changed else "no",
            "yes" if t_changed else "no",
            v_change_dir,
            t_change_dir,
        ]

        # Heuristic answers and correctness
        for name in HEURISTICS:
            row.append(heuristic_answers[name])
        for name in HEURISTICS:
            row.append("correct" if heuristic_correctness[name] else "wrong")

        # Full responses
        row.extend([
            result["vision_responses"][0],
            result["vision_responses"][-1],
            result["text_responses"][0],
            result["text_responses"][-1],
        ])

        results_table.add_data(*row)

        #  Per-step W&B logging (for live charts) ─
        step_log = {
            "step": i + 1,
            "running/vision_r1_accuracy": vision_r1_correct_count / total,
            "running/text_r1_accuracy": text_r1_correct_count / total,
            "running/vision_r2_accuracy": vision_r2_correct_count / total,
            "running/text_r2_accuracy": text_r2_correct_count / total,
            "running/r2_consensus_rate": r2_consensus_count / total,
            "running/vision_change_rate": vision_changed_count / total,
            "running/text_change_rate": text_changed_count / total,
        }
        for name in HEURISTICS:
            step_log[f"running/heuristic_{name}_accuracy"] = heuristic_correct[name] / total

        wandb.log(step_log)

        #  Console output ─
        primary_answer = heuristic_answers["trust_consistent"]
        primary_correct = heuristic_correctness["trust_consistent"]

        print(f"  Vision: {vision_r1}→{vision_r2} ({v_change_dir}) | "
              f"Text: {text_r1}→{text_r2} ({t_change_dir}) | "
              f"GT: {ground_truth} | "
              f"{'✓' if primary_correct else '✗'}")

    #  Aggregate metrics 
    print(f"\n{'='*60}")
    print(f"ThoughtComm Results Summary ({n_test} cases)")
    print(f"{'='*60}")

    print(f"\n  Individual Agent Accuracy:")
    print(f"    Vision R1 (before ThoughtComm): {vision_r1_correct_count}/{total} "
          f"({vision_r1_correct_count/total:.1%})")
    print(f"    Vision R2 (after ThoughtComm):  {vision_r2_correct_count}/{total} "
          f"({vision_r2_correct_count/total:.1%})")
    print(f"    Text R1 (before ThoughtComm):   {text_r1_correct_count}/{total} "
          f"({text_r1_correct_count/total:.1%})")
    print(f"    Text R2 (after ThoughtComm):    {text_r2_correct_count}/{total} "
          f"({text_r2_correct_count/total:.1%})")

    print(f"\n  Consensus:")
    print(f"    R1 consensus rate: {r1_consensus_count}/{total} "
          f"({r1_consensus_count/total:.1%})")
    print(f"    R2 consensus rate: {r2_consensus_count}/{total} "
          f"({r2_consensus_count/total:.1%})")
    if r1_consensus_count > 0:
        print(f"    R1 consensus accuracy: {r1_consensus_correct_count}/{r1_consensus_count} "
              f"({r1_consensus_correct_count/r1_consensus_count:.1%})")
    if r2_consensus_count > 0:
        print(f"    R2 consensus accuracy: {r2_consensus_correct_count}/{r2_consensus_count} "
              f"({r2_consensus_correct_count/r2_consensus_count:.1%})")

    print(f"\n  ThoughtComm Impact (answer changes R1→R2):")
    print(f"    Vision changed: {vision_changed_count}/{total} "
          f"({vision_changed_count/total:.1%})")
    print(f"      Improved (wrong→right): {vision_improved_count}")
    print(f"      Degraded (right→wrong): {vision_degraded_count}")
    print(f"    Text changed: {text_changed_count}/{total} "
          f"({text_changed_count/total:.1%})")
    print(f"      Improved (wrong→right): {text_improved_count}")
    print(f"      Degraded (right→wrong): {text_degraded_count}")

    print(f"\n  Heuristic Comparison:")
    print(f"  {'Heuristic':<25} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'-'*45}")
    for name in HEURISTICS:
        acc = heuristic_correct[name] / total
        print(f"  {name:<25} {acc:>9.1%} {heuristic_correct[name]:>7}/{total}")

    #  W&B summary 
    summary = {
        "results_table": results_table,
        "final/vision_r1_accuracy": vision_r1_correct_count / total,
        "final/vision_r2_accuracy": vision_r2_correct_count / total,
        "final/text_r1_accuracy": text_r1_correct_count / total,
        "final/text_r2_accuracy": text_r2_correct_count / total,
        "final/r1_consensus_rate": r1_consensus_count / total,
        "final/r2_consensus_rate": r2_consensus_count / total,
        "final/r2_consensus_accuracy": (
            r2_consensus_correct_count / r2_consensus_count
            if r2_consensus_count > 0 else 0),
        "final/vision_change_rate": vision_changed_count / total,
        "final/text_change_rate": text_changed_count / total,
        "final/vision_improved": vision_improved_count,
        "final/vision_degraded": vision_degraded_count,
        "final/text_improved": text_improved_count,
        "final/text_degraded": text_degraded_count,
        "final/net_vision_improvement": vision_improved_count - vision_degraded_count,
        "final/net_text_improvement": text_improved_count - text_degraded_count,
        "final/total_cases": total,
    }

    for name in HEURISTICS:
        summary[f"final/heuristic_{name}_accuracy"] = heuristic_correct[name] / total
        summary[f"final/heuristic_{name}_correct"] = heuristic_correct[name]
        if r2_consensus_total > 0:
            summary[f"final/heuristic_{name}_accuracy_consensus"] = (
                heuristic_correct_consensus[name] / r2_consensus_total)
        if r2_disagree_total > 0:
            summary[f"final/heuristic_{name}_accuracy_disagree"] = (
                heuristic_correct_disagree[name] / r2_disagree_total)

    wandb.log(summary)

    for key, val in summary.items():
        if key != "results_table" and isinstance(val, (int, float)):
            wandb.summary[key] = val

    print(f"\n{'='*60}")
    print(f"✓ Results logged to W&B")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    main()