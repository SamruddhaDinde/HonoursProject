"""
Case Study Extraction: Debate Degradation + ThoughtComm Improvement

Extracts specific cases from W&B runs for qualitative analysis:
  F1: Cases where debate DEGRADED the text agent (right R1 → wrong R2)
  F2: Cases where ThoughtComm IMPROVED the text agent (wrong R1 → right R2)

SETUP:
  Option A (W&B API - preferred):
    Fill in DEBATE_RUN_ID and THOUGHTCOMM_RUN_ID below.

  Option B (CSV export):
    1. In W&B: go to Mode 2 debate run → Tables → Download CSV
    2. Go to ThoughtComm run → Tables → Download CSV
    3. Set USE_CSV = True and fill in CSV paths

OUTPUT:
  - Prints the top degradation and improvement cases with full outputs
  - Saves to analysis_case_studies.txt for thesis use
  - Categorises debate degradation cases by failure type (manual step)
"""

import os
import csv
import json

# ── Configuration ────────────────────────────────────────────────────────

USE_CSV = True  # CSV is more reliable; export from W&B UI

# CSV paths (download from W&B UI → Tables → Download CSV)
DEBATE_CSV = "/workspace/analysis/mode2_debate_results.csv"
THOUGHTCOMM_CSV = "/workspace/analysis/thoughtcomm_results.csv"

# W&B run IDs (alternative to CSV)
WANDB_PROJECT = "medical-multiagent"
DEBATE_RUN_ID = "FILL_IN"
THOUGHTCOMM_RUN_ID = "FILL_IN"

# How many cases to extract for each category
N_CASES = 10

OUTPUT_PATH = "/workspace/analysis_case_studies.txt"


# ── CSV Loading ──────────────────────────────────────────────────────────

def load_debate_csv(path: str) -> list:
    """Load Mode 2 debate results from CSV.

    Expected columns (from your evaluator):
      image_id, question, ground_truth, brier_score,
      text_r1_answer, text_r1_correct, vision_r1_answer, vision_r1_correct, r1_agree,
      text_r2_answer, text_r2_correct, vision_r2_answer, vision_r2_correct, r2_agree,
      text_changed, text_change_direction, vision_changed, vision_change_direction,
      meta_answer, meta_correct,
      text_r1_output, vision_r1_output, text_r2_output, vision_r2_output, meta_output,
      ...
    """
    cases = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


def load_thoughtcomm_csv(path: str) -> list:
    """Load ThoughtComm results from CSV.

    Expected columns (from main_thoughtcomm.py):
      image_id, question, ground_truth,
      vision_r1_answer, vision_r1_correct, text_r1_answer, text_r1_correct,
      vision_r2_answer, vision_r2_correct, text_r2_answer, text_r2_correct,
      text_changed, text_change_dir, vision_changed, vision_change_dir,
      ...
    """
    cases = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


# ── Extraction ───────────────────────────────────────────────────────────

def extract_debate_degradation(cases: list) -> list:
    """Find cases where text agent was correct in R1 but wrong in R2.

    These are the persuasion-bias cases: the vision agent's reasoning
    convinced the text agent to abandon a correct answer.
    """
    degraded = []
    for case in cases:
        # Try multiple possible column names
        text_r1_correct = case.get("text_r1_correct", case.get("text_r1_ok", ""))
        text_r2_correct = case.get("text_r2_correct", case.get("text_r2_ok", ""))
        change_dir = case.get("text_change_direction", case.get("text_change_dir", ""))

        is_degraded = False
        if change_dir:
            is_degraded = change_dir.lower().strip() == "right_to_wrong"
        elif text_r1_correct and text_r2_correct:
            is_degraded = (
                text_r1_correct.lower().strip() in ("correct", "true", "1") and
                text_r2_correct.lower().strip() in ("wrong", "false", "0")
            )

        if is_degraded:
            degraded.append(case)

    return degraded


def extract_thoughtcomm_improvement(cases: list) -> list:
    """Find cases where text agent was wrong in R1 but correct in R2.

    These are the ThoughtComm success cases: latent communication
    improved the text agent's diagnosis.
    """
    improved = []
    for case in cases:
        text_r1_correct = case.get("text_r1_correct", case.get("text_r1_ok", ""))
        text_r2_correct = case.get("text_r2_correct", case.get("text_r2_ok", ""))
        change_dir = case.get("text_change_direction", case.get("text_change_dir", ""))

        is_improved = False
        if change_dir:
            is_improved = change_dir.lower().strip() == "wrong_to_right"
        elif text_r1_correct and text_r2_correct:
            is_improved = (
                text_r1_correct.lower().strip() in ("wrong", "false", "0") and
                text_r2_correct.lower().strip() in ("correct", "true", "1")
            )

        if is_improved:
            improved.append(case)

    return improved


# ── Formatting ───────────────────────────────────────────────────────────

def format_debate_case(case: dict, index: int) -> str:
    """Format a debate degradation case for qualitative analysis."""
    lines = [
        f"\n{'='*70}",
        f"DEBATE DEGRADATION CASE #{index}",
        f"{'='*70}",
        f"Image ID: {case.get('image_id', '?')}",
        f"Ground truth: {case.get('ground_truth', '?')}",
        f"Brier score (difficulty): {case.get('brier_score', '?')}",
        f"",
        f"Text R1 answer: {case.get('text_r1_answer', '?')} (CORRECT)",
        f"Text R2 answer: {case.get('text_r2_answer', '?')} (WRONG — degraded)",
        f"Vision R1 answer: {case.get('vision_r1_answer', '?')}",
        f"Vision R2 answer: {case.get('vision_r2_answer', '?')}",
        f"Meta answer: {case.get('meta_answer', case.get('meta_predicted', '?'))}",
        f"",
        f"--- Vision R1 reasoning (what persuaded the text agent) ---",
        f"{case.get('vision_r1_output', case.get('vision_r1_raw', '[not available]'))[:800]}",
        f"",
        f"--- Text R1 reasoning (before seeing vision) ---",
        f"{case.get('text_r1_output', case.get('text_r1_raw', '[not available]'))[:800]}",
        f"",
        f"--- Text R2 reasoning (after seeing vision — now wrong) ---",
        f"{case.get('text_r2_output', case.get('text_r2_raw', '[not available]'))[:800]}",
        f"",
        f"--- YOUR CATEGORISATION (fill in manually) ---",
        f"Failure type: [ ] Hallucinated visual finding",
        f"              [ ] Correct observation, wrong conclusion",
        f"              [ ] Authoritative but vague",
        f"              [ ] Specific detail that overrode text reasoning",
        f"              [ ] Other: _______________",
        f"Notes: ",
    ]
    return "\n".join(lines)


def format_thoughtcomm_case(case: dict, index: int) -> str:
    """Format a ThoughtComm improvement case for qualitative analysis."""
    lines = [
        f"\n{'='*70}",
        f"THOUGHTCOMM IMPROVEMENT CASE #{index}",
        f"{'='*70}",
        f"Image ID: {case.get('image_id', '?')}",
        f"Ground truth: {case.get('ground_truth', '?')}",
        f"",
        f"Text R1 answer: {case.get('text_r1_answer', '?')} (WRONG)",
        f"Text R2 answer: {case.get('text_r2_answer', '?')} (CORRECT — improved)",
        f"Vision R1 answer: {case.get('vision_r1_answer', '?')}",
        f"Vision R2 answer: {case.get('vision_r2_answer', '?')}",
        f"",
        f"--- Text R1 reasoning (before ThoughtComm) ---",
        f"{case.get('text_r1_output', case.get('text_r1_response', '[not available]'))[:800]}",
        f"",
        f"--- Text R2 reasoning (after ThoughtComm — now correct) ---",
        f"{case.get('text_r2_output', case.get('text_r2_response', '[not available]'))[:800]}",
        f"",
        f"--- What changed? (fill in manually) ---",
        f"Did R2 mention new visual information? [ ] Yes  [ ] No",
        f"Did R2 shift emphasis to different clinical features? [ ] Yes  [ ] No",
        f"Did R2 reconsider a differential? [ ] Yes  [ ] No",
        f"Notes: ",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    output_lines = []
    output_lines.append("CASE STUDY EXTRACTION FOR THESIS ANALYSIS")
    output_lines.append(f"{'='*70}\n")

    # ── F1: Debate degradation cases ──────────────────────────────────
    print("Loading debate results...")
    if USE_CSV:
        if not os.path.exists(DEBATE_CSV):
            print(f"  CSV not found at {DEBATE_CSV}")
            print(f"  Export from W&B: go to Mode 2 debate run → Tables → Download CSV")
            print(f"  Save to {DEBATE_CSV}")
            debate_cases = []
        else:
            debate_cases = load_debate_csv(DEBATE_CSV)
            print(f"  Loaded {len(debate_cases)} cases")
    else:
        print("  W&B API loading not implemented — use CSV export")
        debate_cases = []

    if debate_cases:
        degraded = extract_debate_degradation(debate_cases)
        print(f"  Found {len(degraded)} debate degradation cases (text right→wrong)")

        output_lines.append(f"SECTION F1: DEBATE DEGRADATION CASES")
        output_lines.append(f"Found {len(degraded)} cases where text agent was correct in R1 "
                          f"but wrong in R2 after seeing vision agent's reasoning.\n")

        for i, case in enumerate(degraded[:N_CASES], 1):
            formatted = format_debate_case(case, i)
            output_lines.append(formatted)
            print(f"  Extracted case #{i}: image_id={case.get('image_id', '?')}")

    # ── F2: ThoughtComm improvement cases ─────────────────────────────
    print("\nLoading ThoughtComm results...")
    if USE_CSV:
        if not os.path.exists(THOUGHTCOMM_CSV):
            print(f"  CSV not found at {THOUGHTCOMM_CSV}")
            print(f"  Export from W&B: go to ThoughtComm run → Tables → Download CSV")
            print(f"  Save to {THOUGHTCOMM_CSV}")
            tc_cases = []
        else:
            tc_cases = load_thoughtcomm_csv(THOUGHTCOMM_CSV)
            print(f"  Loaded {len(tc_cases)} cases")
    else:
        tc_cases = []

    if tc_cases:
        improved = extract_thoughtcomm_improvement(tc_cases)
        print(f"  Found {len(improved)} ThoughtComm improvement cases (text wrong→right)")

        output_lines.append(f"\n\n{'='*70}")
        output_lines.append(f"SECTION F2: THOUGHTCOMM IMPROVEMENT CASES")
        output_lines.append(f"Found {len(improved)} cases where text agent was wrong in R1 "
                          f"but correct in R2 after latent communication.\n")

        for i, case in enumerate(improved[:N_CASES], 1):
            formatted = format_thoughtcomm_case(case, i)
            output_lines.append(formatted)
            print(f"  Extracted case #{i}: image_id={case.get('image_id', '?')}")

    # ── Save ──────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nCase studies saved to {OUTPUT_PATH}")
    print(f"\nNEXT STEP: Open {OUTPUT_PATH} and fill in the categorisation fields")
    print(f"for each case. This is the qualitative analysis your supervisor wants.")


if __name__ == "__main__":
    main()
