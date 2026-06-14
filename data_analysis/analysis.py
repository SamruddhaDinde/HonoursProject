"""
Comprehensive thesis data analysis.

Reads:
  - The NEJM dataset JSON (for brier_score, public votes, case metadata)
  - W&B table exports (CSV) for each experiment

Produces:
  1. Difficulty stratification (easy/medium/hard) × experiment accuracy
  2. Oracle ceiling per difficulty tier
  3. Cross-experiment per-case consistency heatmap
  4. Public vote vs agent accuracy correlation
  5. Confidence calibration curves (Mode 3, LangGraph experiments)
  6. Full 2×2 transition matrices (debate/LangGraph experiments)
  7. Agreement-conditioned accuracy across all experiments

Usage:
    python analysis.py --dataset path/to/nejm_cases.json --wandb-dir path/to/exports/

    The --wandb-dir should contain CSV exports from W&B tables, one per run.
    Export from W&B: go to each run → Artifacts → results_table → download CSV.
    Name them descriptively (e.g., mode1_689.csv, conditional_debate_689.csv).

    Alternatively, use the W&B API to pull tables directly (see pull_from_wandb_api below).
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD THE NEJM DATASET
# ──────────────────────────────────────────────────────────────────────

def parse_int(value, default=0):
    """Safely parse integers from messy dataset fields.

    Handles values such as:
      - "85230."
      - "165,436"
      - "85,230."
      - 123
      - None
    """
    if value is None:
        return default

    s = str(value).strip()

    if s == "" or s.lower() in {"none", "nan", "null"}:
        return default

    # Remove common formatting artifacts.
    s = s.replace(",", "")
    s = s.replace("%", "")

    # Handle values like "85230."
    if s.endswith("."):
        s = s[:-1]

    try:
        return int(float(s))
    except ValueError:
        return default


def parse_percentage(value, default=0.0):
    """Safely parse percentage fields such as '53%', '53.2%', or 53.2.

    Returns a proportion between 0 and 1.
    """
    if value is None:
        return default

    s = str(value).strip()

    if s == "" or s.lower() in {"none", "nan", "null"}:
        return default

    s = s.replace(",", "")

    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        number = float(s)
        # If the dataset ever stores 53 instead of 0.53, treat it as 53%.
        return number / 100.0 if number > 1 else number
    except ValueError:
        return default


def parse_bool(value):
    """Safely parse correctness values from W&B CSV exports.

    Handles values such as:
      - True / False
      - "true" / "false"
      - "correct" / "incorrect"
      - 1 / 0
      - "yes" / "no"
      - "ok" / "x"
    Returns True, False, or np.nan when the value is unparseable.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False

    s = str(value).strip().lower()
    s = s.strip('"\'')

    true_values = {
        "true", "t", "1", "yes", "y", "ok", "okay",
        "correct", "right", "success", "successful",
        "✅", "✓", "✔",
    }
    false_values = {
        "false", "f", "0", "no", "n",
        "incorrect", "wrong", "fail", "failed", "failure",
        "❌", "✗", "x",
    }

    if s in true_values:
        return True
    if s in false_values:
        return False

    return np.nan


def normalize_correct_series(series, column_name="correct", experiment_name=""):
    """Convert a correctness column to real booleans so accuracy means work."""
    converted = series.apply(parse_bool)
    unknown_mask = converted.isna()

    if unknown_mask.any():
        examples = (
            series[unknown_mask]
            .dropna()
            .astype(str)
            .unique()[:5]
        )
        location = f"{experiment_name}:{column_name}" if experiment_name else column_name
        print(
            f"    Warning: {location} has {unknown_mask.sum()} unparseable correctness "
            f"values; treating them as False. Examples: {list(examples)}"
        )

    return converted.fillna(False).astype(bool)


def load_dataset(path):
    """Load the NEJM JSON and compute per-case difficulty metrics."""
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)

    rows = []
    for c in cases:
        # Parse public vote for the correct answer
        correct_key = f"option_{c['correct_answer']}_votes"
        correct_pct = parse_percentage(c.get(correct_key, "0%"))

        rows.append({
            "image_id": str(c["image_id"]),
            "correct_answer": c["correct_answer"],
            "brier_score": c.get("brier_score", None),
            "public_correct_pct": correct_pct,
            "total_votes": parse_int(c.get("total_number_votes", 0)),
            "question": c.get("question", ""),
        })

    df = pd.DataFrame(rows)
    df["brier_score"] = pd.to_numeric(df["brier_score"], errors="coerce")

    # Difficulty tiers based on public correctness rate
    # Easy: >60% of public got it right; Medium: 30-60%; Hard: <30%
    df["difficulty"] = pd.cut(
        df["public_correct_pct"],
        bins=[0, 0.30, 0.60, 1.01],
        labels=["hard", "medium", "easy"],
        include_lowest=True,
    )

    # Alternative: difficulty from brier score (lower = easier for public)
    if df["brier_score"].notna().sum() > 0:
        df["difficulty_brier"] = pd.cut(
            df["brier_score"],
            bins=[0, 0.15, 0.40, 1.01],
            labels=["easy", "medium", "hard"],
            include_lowest=True,
        )

    print(f"Loaded {len(df)} cases. Difficulty distribution (by public vote):")
    print(df["difficulty"].value_counts().sort_index())
    print()
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. LOAD W&B EXPERIMENT TABLES
# ──────────────────────────────────────────────────────────────────────

def load_experiment_csv(csv_path, experiment_name):
    """Load a single W&B export CSV. Expects at minimum: image_id, ground_truth, and some answer columns."""
    df = pd.read_csv(csv_path)
    df["experiment"] = experiment_name

    # Standardise column names (different experiments use slightly different names)
    renames = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ("meta_predicted", "final_answer", "predicted"):
            renames[col] = "final_answer"
        if lc in ("meta_correct", "final_correct", "correct"):
            renames[col] = "final_correct"
        if lc in ("text_predicted", "text_initial", "text_top"):
            renames[col] = "text_answer"
        if lc in ("text_correct", "text_initial_correct"):
            renames[col] = "text_correct"
        if lc in ("vision_predicted", "vision_answer", "vision_top"):
            renames[col] = "vision_answer"
        if lc in ("vision_correct",):
            renames[col] = "vision_correct"
    df.rename(columns=renames, inplace=True)

    # Ensure correctness columns are real booleans.
    # W&B CSV exports may store these as True/False, true/false, correct/incorrect,
    # or even pandas string dtype, so dtype == object is not sufficient.
    for col in ["final_correct", "text_correct", "vision_correct"]:
        if col in df.columns:
            df[col] = normalize_correct_series(
                df[col], column_name=col, experiment_name=experiment_name
            )

    # Make image_id join keys consistent across dataset JSON and W&B CSVs.
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str)

    return df


def load_all_experiments(wandb_dir):
    """Load all CSV files from the W&B export directory."""
    experiments = {}
    wandb_path = Path(wandb_dir)
    for csv_file in sorted(wandb_path.glob("*.csv")):
        name = csv_file.stem  # filename without extension as experiment name
        print(f"  Loading {name} from {csv_file.name}...")
        experiments[name] = load_experiment_csv(csv_file, name)
    print(f"\nLoaded {len(experiments)} experiment tables.\n")
    return experiments


# ──────────────────────────────────────────────────────────────────────
# 3. DIFFICULTY STRATIFICATION
# ──────────────────────────────────────────────────────────────────────

def difficulty_stratification(dataset_df, experiments):
    """Compute per-experiment accuracy broken down by difficulty tier."""
    print("=" * 70)
    print("ANALYSIS 1: DIFFICULTY STRATIFICATION")
    print("=" * 70)

    results = []
    for name, exp_df in experiments.items():
        if "final_correct" not in exp_df.columns:
            continue

        merged = exp_df.merge(dataset_df[["image_id", "difficulty", "public_correct_pct"]],
                              on="image_id", how="left")

        for tier in ["easy", "medium", "hard"]:
            subset = merged[merged["difficulty"] == tier]
            if len(subset) == 0:
                continue
            acc = subset["final_correct"].mean()
            results.append({
                "experiment": name,
                "difficulty": tier,
                "accuracy": round(acc * 100, 2),
                "n_cases": len(subset),
            })

        # Overall
        acc = merged["final_correct"].mean()
        results.append({
            "experiment": name,
            "difficulty": "all",
            "accuracy": round(acc * 100, 2),
            "n_cases": len(merged),
        })

    result_df = pd.DataFrame(results)
    pivot = result_df.pivot_table(
        index="experiment", columns="difficulty", values="accuracy", aggfunc="first"
    )
    pivot = pivot.reindex(columns=["easy", "medium", "hard", "all"])
    print(pivot.to_string())
    print()
    pivot.to_csv("analysis_difficulty_stratification.csv")
    return result_df


# ──────────────────────────────────────────────────────────────────────
# 4. ORACLE CEILING PER DIFFICULTY TIER
# ──────────────────────────────────────────────────────────────────────

def oracle_by_difficulty(dataset_df, experiments):
    """Compute oracle ceiling (at least one specialist correct) per difficulty tier.
    Uses experiments that have both text_correct and vision_correct columns."""
    print("=" * 70)
    print("ANALYSIS 2: ORACLE CEILING PER DIFFICULTY TIER")
    print("=" * 70)

    results = []
    for name, exp_df in experiments.items():
        if "text_correct" not in exp_df.columns or "vision_correct" not in exp_df.columns:
            continue

        merged = exp_df.merge(dataset_df[["image_id", "difficulty"]], on="image_id", how="left")
        merged["either_correct"] = merged["text_correct"] | merged["vision_correct"]

        for tier in ["easy", "medium", "hard", "all"]:
            subset = merged if tier == "all" else merged[merged["difficulty"] == tier]
            if len(subset) == 0:
                continue
            oracle = subset["either_correct"].mean()
            results.append({
                "experiment": name,
                "difficulty": tier,
                "oracle_pct": round(oracle * 100, 2),
                "n_cases": len(subset),
                "n_either_correct": int(subset["either_correct"].sum()),
            })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        pivot = result_df.pivot_table(index="experiment", columns="difficulty",
                                       values="oracle_pct", aggfunc="first")
        pivot = pivot.reindex(columns=["easy", "medium", "hard", "all"])
        print(pivot.to_string())
        pivot.to_csv("analysis_oracle_by_difficulty.csv")
    print()
    return result_df


# ──────────────────────────────────────────────────────────────────────
# 5. CROSS-EXPERIMENT PER-CASE CONSISTENCY
# ──────────────────────────────────────────────────────────────────────

def cross_experiment_consistency(experiments):
    """For each case, count how many experiments got it right. Identifies
    universally-hard and universally-easy cases."""
    print("=" * 70)
    print("ANALYSIS 3: CROSS-EXPERIMENT CASE CONSISTENCY")
    print("=" * 70)

    case_results = defaultdict(dict)
    exp_names = []
    for name, exp_df in experiments.items():
        if "final_correct" not in exp_df.columns or "image_id" not in exp_df.columns:
            continue
        exp_names.append(name)
        for _, row in exp_df.iterrows():
            case_results[row["image_id"]][name] = bool(row["final_correct"])

    if not case_results:
        print("No compatible experiments found.\n")
        return None

    # Build a case × experiment matrix
    case_ids = sorted(case_results.keys())
    matrix = pd.DataFrame(index=case_ids, columns=exp_names, dtype=float)
    for cid in case_ids:
        for exp in exp_names:
            matrix.loc[cid, exp] = 1.0 if case_results[cid].get(exp, False) else 0.0

    matrix["n_correct"] = matrix.sum(axis=1).astype(int)
    matrix["pct_correct"] = (matrix["n_correct"] / len(exp_names) * 100).round(1)

    # Summary
    n_exp = len(exp_names)
    always_right = (matrix["n_correct"] == n_exp).sum()
    always_wrong = (matrix["n_correct"] == 0).sum()
    mixed = len(matrix) - always_right - always_wrong
    print(f"  {len(exp_names)} experiments, {len(case_ids)} cases")
    print(f"  Always correct (all {n_exp}): {always_right} cases")
    print(f"  Always wrong (0/{n_exp}):     {always_wrong} cases")
    print(f"  Mixed:                        {mixed} cases")
    print(f"  Mean cross-experiment agreement: {matrix['pct_correct'].mean():.1f}%")
    print()

    # Correlation between experiments (do they find the same cases hard?)
    exp_matrix = matrix[exp_names].astype(float)
    corr = exp_matrix.corr()
    print("Inter-experiment correlation (are the same cases hard/easy?):")
    print(corr.round(3).to_string())
    print()

    matrix.to_csv("analysis_case_consistency.csv")
    corr.to_csv("analysis_experiment_correlation.csv")
    return matrix


# ──────────────────────────────────────────────────────────────────────
# 6. PUBLIC VOTE CORRELATION
# ──────────────────────────────────────────────────────────────────────

def public_vote_correlation(dataset_df, experiments):
    """Correlate public correctness rate with each experiment's per-case accuracy."""
    print("=" * 70)
    print("ANALYSIS 4: PUBLIC VOTE VS AGENT ACCURACY CORRELATION")
    print("=" * 70)

    results = []
    for name, exp_df in experiments.items():
        if "final_correct" not in exp_df.columns:
            continue
        merged = exp_df.merge(dataset_df[["image_id", "public_correct_pct"]], on="image_id", how="left")
        merged = merged.dropna(subset=["public_correct_pct"])
        if len(merged) < 10:
            continue

        corr = merged["public_correct_pct"].corr(merged["final_correct"].astype(float))
        results.append({"experiment": name, "correlation_with_public": round(corr, 4), "n": len(merged)})
        print(f"  {name}: r = {corr:.4f} (n={len(merged)})")

    print()
    result_df = pd.DataFrame(results)
    result_df.to_csv("analysis_public_vote_correlation.csv", index=False)
    return result_df


# ──────────────────────────────────────────────────────────────────────
# 7. CONFIDENCE CALIBRATION
# ──────────────────────────────────────────────────────────────────────

def confidence_calibration(experiments):
    """For experiments with confidence columns, bucket by confidence and compute
    accuracy per bucket to assess calibration."""
    print("=" * 70)
    print("ANALYSIS 5: CONFIDENCE CALIBRATION")
    print("=" * 70)

    for name, exp_df in experiments.items():
        # Look for text confidence columns
        conf_col = None
        correct_col = None
        for c in exp_df.columns:
            lc = c.lower()
            if "text_confidence" in lc or "confidence" in lc:
                conf_col = c
            if "text_correct" in lc or "final_correct" in lc:
                correct_col = c

        if conf_col is None or correct_col is None:
            continue

        df = exp_df[[conf_col, correct_col]].copy()
        df.columns = ["confidence", "correct"]
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df["correct"] = normalize_correct_series(
            df["correct"], column_name=correct_col, experiment_name=name
        )
        df = df.dropna(subset=["confidence"])

        if len(df) < 20:
            continue

        # Determine if confidence is 0-1 scale or 1-5 scale
        max_conf = df["confidence"].max()
        if max_conf > 1:
            # 1-5 scale: use integer bins
            bins = [0, 1, 2, 3, 4, 5]
            labels = ["1", "2", "3", "4", "5"]
        else:
            # 0-1 scale: use quintile bins
            bins = [0, 0.5, 0.7, 0.8, 0.9, 1.01]
            labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

        df["conf_bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, include_lowest=True)

        cal = df.groupby("conf_bucket", observed=True).agg(
            accuracy=("correct", "mean"),
            n_cases=("correct", "count"),
        ).reset_index()
        cal["accuracy"] = (cal["accuracy"] * 100).round(2)

        print(f"\n  {name} (confidence column: {conf_col}):")
        print(cal.to_string(index=False))
        cal.to_csv(f"analysis_calibration_{name}.csv", index=False)
    print()


# ──────────────────────────────────────────────────────────────────────
# 8. FULL TRANSITION MATRICES
# ──────────────────────────────────────────────────────────────────────

def transition_matrices(experiments):
    """For experiments with text_initial and final_answer, compute the full
    2×2 transition matrix: text_correct × final_correct."""
    print("=" * 70)
    print("ANALYSIS 6: TRANSITION MATRICES (text_initial → final)")
    print("=" * 70)

    for name, exp_df in experiments.items():
        # Need text correctness and final correctness
        text_col = None
        final_col = None
        for c in exp_df.columns:
            lc = c.lower().strip()
            if lc in ("text_correct", "text_initial_correct"):
                text_col = c
            if lc in ("final_correct", "meta_correct"):
                final_col = c

        if text_col is None or final_col is None:
            continue

        df = exp_df[[text_col, final_col]].copy()
        df.columns = ["text_correct", "final_correct"]

        # Ensure boolean, even if the CSV used string dtype or True/False text.
        df["text_correct"] = normalize_correct_series(
            df["text_correct"], column_name=text_col, experiment_name=name
        )
        df["final_correct"] = normalize_correct_series(
            df["final_correct"], column_name=final_col, experiment_name=name
        )

        # 2×2 matrix
        rr = ((df["text_correct"]) & (df["final_correct"])).sum()       # right → right
        rw = ((df["text_correct"]) & (~df["final_correct"])).sum()      # right → wrong (hurt)
        wr = ((~df["text_correct"]) & (df["final_correct"])).sum()      # wrong → right (helped)
        ww = ((~df["text_correct"]) & (~df["final_correct"])).sum()     # wrong → wrong

        total = len(df)
        print(f"\n  {name} ({total} cases):")
        print(f"    text_right → final_right (preserved): {rr:4d}  ({rr/total*100:5.1f}%)")
        print(f"    text_right → final_wrong (HURT):      {rw:4d}  ({rw/total*100:5.1f}%)")
        print(f"    text_wrong → final_right (HELPED):    {wr:4d}  ({wr/total*100:5.1f}%)")
        print(f"    text_wrong → final_wrong (stuck):     {ww:4d}  ({ww/total*100:5.1f}%)")
        print(f"    Net: helped {wr} - hurt {rw} = {wr - rw:+d}")

        matrix_df = pd.DataFrame({
            "final_correct": [rr, wr],
            "final_wrong": [rw, ww],
        }, index=["text_correct", "text_wrong"])
        matrix_df.to_csv(f"analysis_transition_{name}.csv")
    print()


# ──────────────────────────────────────────────────────────────────────
# 9. AGREEMENT-CONDITIONED ACCURACY (UNIFIED)
# ──────────────────────────────────────────────────────────────────────

def agreement_analysis(experiments):
    """Compute accuracy when agents agree vs disagree, across all experiments
    that have both text_answer and vision_answer."""
    print("=" * 70)
    print("ANALYSIS 7: AGREEMENT-CONDITIONED ACCURACY")
    print("=" * 70)

    results = []
    for name, exp_df in experiments.items():
        if "text_answer" not in exp_df.columns or "vision_answer" not in exp_df.columns:
            # Try alternate column names
            text_col = next((c for c in exp_df.columns if "text_predicted" in c.lower()), None)
            vision_col = next((c for c in exp_df.columns if "vision_predicted" in c.lower()), None)
            if text_col and vision_col:
                exp_df = exp_df.rename(columns={text_col: "text_answer", vision_col: "vision_answer"})
            else:
                continue

        if "final_correct" not in exp_df.columns:
            continue

        df = exp_df[["text_answer", "vision_answer", "final_correct"]].copy()
        df["final_correct"] = normalize_correct_series(
            df["final_correct"], column_name="final_correct", experiment_name=name
        )
        df["agree"] = (
            df["text_answer"].astype("string").str.strip().str.upper()
            == df["vision_answer"].astype("string").str.strip().str.upper()
        )

        agree_df = df[df["agree"]]
        disagree_df = df[~df["agree"]]

        agree_acc = agree_df["final_correct"].mean() * 100 if len(agree_df) > 0 else 0
        disagree_acc = disagree_df["final_correct"].mean() * 100 if len(disagree_df) > 0 else 0

        results.append({
            "experiment": name,
            "agree_n": len(agree_df),
            "agree_accuracy": round(agree_acc, 2),
            "disagree_n": len(disagree_df),
            "disagree_accuracy": round(disagree_acc, 2),
            "agreement_rate": round(len(agree_df) / len(df) * 100, 1),
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        print(result_df.to_string(index=False))
        result_df.to_csv("analysis_agreement_conditioned.csv", index=False)
    print()
    return result_df


# ──────────────────────────────────────────────────────────────────────
# BONUS: PULL TABLES FROM W&B API (if CSVs not yet exported)
# ──────────────────────────────────────────────────────────────────────

def pull_from_wandb_api(entity, project, output_dir="wandb_exports"):
    """Pull all results_table artifacts from a W&B project as CSVs.
    Requires: pip install wandb"""
    import wandb
    api = wandb.Api()
    os.makedirs(output_dir, exist_ok=True)

    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        try:
            for artifact in run.logged_artifacts():
                if "results_table" in artifact.name:
                    table = artifact.get("results_table")
                    if table is not None:
                        df = pd.DataFrame(data=table.data, columns=table.columns)
                        fname = f"{output_dir}/{run.name}.csv"
                        df.to_csv(fname, index=False)
                        print(f"  Exported {run.name} → {fname} ({len(df)} rows)")
        except Exception as e:
            # Try getting table from run summary instead
            try:
                for key, val in run.summary.items():
                    if "table" in key.lower() and hasattr(val, "get"):
                        table_ref = val
                        # This is a table reference; download via artifact
                        pass
            except Exception:
                pass
            print(f"  Skipped {run.name}: {e}")

    print(f"\nExported to {output_dir}/. Use --wandb-dir {output_dir} to analyse.")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive thesis data analysis")
    parser.add_argument("--dataset", required=True, help="Path to NEJM JSON file")
    parser.add_argument("--wandb-dir", required=True, help="Directory containing W&B CSV exports")
    parser.add_argument("--pull-wandb", action="store_true",
                        help="Pull tables from W&B API first (requires wandb login)")
    parser.add_argument("--entity", default="samruddha-dinde-university-of-technology-sydney")
    parser.add_argument("--project", default="medical-multiagent")
    args = parser.parse_args()

    if args.pull_wandb:
        pull_from_wandb_api(args.entity, args.project, output_dir=args.wandb_dir)

    # Load data
    print("Loading NEJM dataset...")
    dataset_df = load_dataset(args.dataset)
    print("Loading W&B experiment tables...")
    experiments = load_all_experiments(args.wandb_dir)

    if not experiments:
        print("No experiment CSVs found. Export tables from W&B or use --pull-wandb.")
        return

    # Run all analyses
    difficulty_stratification(dataset_df, experiments)
    oracle_by_difficulty(dataset_df, experiments)
    cross_experiment_consistency(experiments)
    public_vote_correlation(dataset_df, experiments)
    confidence_calibration(experiments)
    transition_matrices(experiments)
    agreement_analysis(experiments)

    print("=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    for f in sorted(Path(".").glob("analysis_*.csv")):
        print(f"  {f}")
    print("\nUse these CSVs to build thesis tables and figures.")


if __name__ == "__main__":
    main()