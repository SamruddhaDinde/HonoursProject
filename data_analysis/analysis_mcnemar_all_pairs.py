"""
Pairwise McNemar Analysis for Thesis Experiments

This script auto-loads every CSV in your W&B export directory, detects the
per-case correctness column, and runs exact paired McNemar tests for all
compatible experiment pairs.

Run from project root:

    python data_analysis/analysis_mcnemar_all_pairs.py --csv-dir wandb_exports

Useful options:

    python data_analysis/analysis_mcnemar_all_pairs.py --csv-dir wandb_exports --min-common 600
    python data_analysis/analysis_mcnemar_all_pairs.py --csv-dir wandb_exports --n-boot 10000

Outputs:
  - analysis_mcnemar_all_pairs.txt
  - analysis_mcnemar_all_pairs.csv

Notes:
  - McNemar compares paired binary outcomes on the intersection of image_id values.
  - If one experiment has 689 cases and another has 339, the comparison uses only
    their common cases.
  - P-values are unadjusted by default. The CSV includes Bonferroni and
    Benjamini-Hochberg/FDR adjusted p-values as well.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# ──────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────

TRUE_VALUES = {
    "true", "t", "1", "yes", "y", "ok", "okay",
    "correct", "right", "success", "successful",
    "✅", "✓", "✔",
}

FALSE_VALUES = {
    "false", "f", "0", "no", "n", "x",
    "incorrect", "wrong", "fail", "failed", "failure",
    "❌", "✗", "✘",
}


def parse_bool(value) -> Optional[bool]:
    """Parse common correctness values into True/False."""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return bool(value)

    s = str(value).strip().lower()
    s = s.strip('"').strip("'").strip()

    if s in {"", "none", "nan", "null"}:
        return None

    if s in TRUE_VALUES:
        return True
    if s in FALSE_VALUES:
        return False

    s_clean = s.rstrip(".,;:! ")
    if s_clean in TRUE_VALUES:
        return True
    if s_clean in FALSE_VALUES:
        return False

    return None


def normalize_image_id(value) -> str:
    """Normalise image IDs so 123 and 123.0 match."""
    s = str(value).strip()

    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def clean_experiment_name(csv_path: Path) -> str:
    """Create a readable experiment name from a CSV filename."""
    name = csv_path.stem

    # Handle accidental double extension, e.g. thoughtcomm_339.csv.csv
    if name.endswith(".csv"):
        name = name[:-4]

    return name


# ──────────────────────────────────────────────────────────────────────
# Column detection
# ──────────────────────────────────────────────────────────────────────

ID_CANDIDATES = [
    "image_id",
    "case_id",
    "id",
    "question_id",
    "case",
]

CORRECTNESS_PRIORITY = [
    # Most common final / meta columns
    "final_correct",
    "meta_correct",
    "correct",
    "is_correct",
    "answer_correct",
    "prediction_correct",

    # Single-agent / named variants
    "model_correct",
    "agent_correct",
    "predicted_correct",

    # Use these only if no final/meta column exists
    "text_correct",
    "vision_correct",
]


def find_column_case_insensitive(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_to_original = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lower_to_original:
            return lower_to_original[key]
    return None


def detect_correctness_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the best correctness column.

    First tries known names. If not found, tries any column ending in "_correct"
    that has parseable boolean-like values.
    """
    columns = list(df.columns)

    preferred = find_column_case_insensitive(columns, CORRECTNESS_PRIORITY)
    if preferred is not None:
        return preferred

    # Fallback: any column containing "correct" with mostly parseable values.
    possible = [c for c in columns if "correct" in c.lower()]

    best_col = None
    best_parse_rate = 0.0

    for col in possible:
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue

        parsed = sample.map(parse_bool)
        parse_rate = parsed.notna().mean()

        if parse_rate > best_parse_rate:
            best_parse_rate = parse_rate
            best_col = col

    if best_col is not None and best_parse_rate >= 0.8:
        return best_col

    return None


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_experiment_csv(csv_path: Path) -> Tuple[str, Dict[str, bool], dict]:
    """Load one experiment CSV into {image_id: bool}."""
    name = clean_experiment_name(csv_path)

    # utf-8-sig handles BOM; errors=replace avoids Windows cp1252 crashes.
    df = pd.read_csv(csv_path, encoding="utf-8-sig", encoding_errors="replace")

    id_col = find_column_case_insensitive(list(df.columns), ID_CANDIDATES)
    correct_col = detect_correctness_column(df)

    metadata = {
        "name": name,
        "path": str(csv_path),
        "rows": len(df),
        "id_col": id_col,
        "correct_col": correct_col,
        "loaded_cases": 0,
        "skipped_rows": 0,
        "accuracy": np.nan,
    }

    if id_col is None:
        metadata["error"] = f"No image_id column found. Columns: {list(df.columns)}"
        return name, {}, metadata

    if correct_col is None:
        metadata["error"] = f"No correctness column found. Columns: {list(df.columns)}"
        return name, {}, metadata

    results: Dict[str, bool] = {}
    skipped = 0

    for _, row in df.iterrows():
        img_id = normalize_image_id(row[id_col])
        parsed = parse_bool(row[correct_col])

        if not img_id or parsed is None:
            skipped += 1
            continue

        results[img_id] = parsed

    metadata["loaded_cases"] = len(results)
    metadata["skipped_rows"] = skipped
    metadata["accuracy"] = float(np.mean(list(results.values()))) if results else np.nan

    return name, results, metadata


def load_all_experiments(csv_dir: Path) -> Tuple[Dict[str, Dict[str, bool]], List[dict]]:
    csv_files = sorted(csv_dir.glob("*.csv"))

    all_results: Dict[str, Dict[str, bool]] = {}
    metadata_rows: List[dict] = []

    print(f"Found {len(csv_files)} CSV files in {csv_dir}")

    for path in csv_files:
        name, results, metadata = load_experiment_csv(path)
        metadata_rows.append(metadata)

        if results:
            # Ensure unique names if two files have same stem.
            final_name = name
            suffix = 2
            while final_name in all_results:
                final_name = f"{name}_{suffix}"
                suffix += 1

            all_results[final_name] = results

            print(
                f"  Loaded {final_name}: {len(results)} cases, "
                f"accuracy={metadata['accuracy']:.1%}, "
                f"id={metadata['id_col']!r}, correct={metadata['correct_col']!r}"
            )
        else:
            print(f"  SKIPPED {name}: {metadata.get('error', 'no usable rows')}")

    return all_results, metadata_rows


# ──────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────

def mcnemar_exact(results_a: Dict[str, bool], results_b: Dict[str, bool]) -> dict:
    common_ids = sorted(set(results_a) & set(results_b))
    n = len(common_ids)

    if n == 0:
        return {
            "n_common": 0,
            "p_value": np.nan,
            "both_right": 0,
            "a_only": 0,
            "b_only": 0,
            "both_wrong": 0,
            "discordant": 0,
            "acc_a": np.nan,
            "acc_b": np.nan,
            "diff": np.nan,
            "diff_pp": np.nan,
        }

    both_right = a_only = b_only = both_wrong = 0

    for img_id in common_ids:
        a_ok = results_a[img_id]
        b_ok = results_b[img_id]

        if a_ok and b_ok:
            both_right += 1
        elif a_ok and not b_ok:
            a_only += 1
        elif not a_ok and b_ok:
            b_only += 1
        else:
            both_wrong += 1

    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            k=min(a_only, b_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    acc_a = (both_right + a_only) / n
    acc_b = (both_right + b_only) / n
    diff = acc_b - acc_a

    return {
        "n_common": n,
        "p_value": p_value,
        "both_right": both_right,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "diff": diff,
        "diff_pp": diff * 100,
    }


def bootstrap_ci(
    results_a: Dict[str, bool],
    results_b: Dict[str, bool],
    n_boot: int,
    seed: int = 42,
) -> Tuple[float, float, float]:
    common_ids = sorted(set(results_a) & set(results_b))
    n = len(common_ids)

    if n == 0:
        return np.nan, np.nan, np.nan

    a = np.array([int(results_a[i]) for i in common_ids])
    b = np.array([int(results_b[i]) for i in common_ids])

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = b[idx].mean() - a[idx].mean()

    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)
    mean = diffs.mean()

    return lower, upper, mean


def bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    return [min(p * m, 1.0) if not np.isnan(p) else np.nan for p in p_values]


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Return BH/FDR adjusted p-values in original order."""
    p = np.array(p_values, dtype=float)
    m = len(p)

    adjusted = np.full(m, np.nan)
    valid = ~np.isnan(p)

    if valid.sum() == 0:
        return adjusted.tolist()

    valid_indices = np.where(valid)[0]
    valid_p = p[valid]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    adjusted_sorted = np.empty_like(sorted_p)
    prev = 1.0

    for i in range(len(sorted_p) - 1, -1, -1):
        rank = i + 1
        value = sorted_p[i] * len(sorted_p) / rank
        prev = min(prev, value)
        adjusted_sorted[i] = min(prev, 1.0)

    adjusted_valid = np.empty_like(valid_p)
    adjusted_valid[order] = adjusted_sorted
    adjusted[valid_indices] = adjusted_valid

    return adjusted.tolist()


def significance(p: float) -> str:
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="wandb_exports")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument(
        "--min-common",
        type=int,
        default=1,
        help="Minimum number of common cases required for a comparison. Use 600 to restrict mostly to 689-case runs.",
    )
    parser.add_argument("--output-prefix", default="analysis_mcnemar_all_pairs")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)

    print("=" * 70)
    print("All-Pairs McNemar Analysis")
    print("=" * 70)

    all_results, metadata_rows = load_all_experiments(csv_dir)

    if len(all_results) < 2:
        print("Need at least two experiments with usable data.")
        return

    print(f"\nRunning all pairwise comparisons among {len(all_results)} experiments...")
    print(f"Minimum common cases: {args.min_common}")
    print(f"Bootstrap samples: {args.n_boot}")

    rows = []
    experiment_names = sorted(all_results.keys())

    for name_a, name_b in itertools.combinations(experiment_names, 2):
        results_a = all_results[name_a]
        results_b = all_results[name_b]

        result = mcnemar_exact(results_a, results_b)

        if result["n_common"] < args.min_common:
            continue

        ci_lower, ci_upper, ci_mean = bootstrap_ci(
            results_a,
            results_b,
            n_boot=args.n_boot,
        )

        rows.append({
            "comparison": f"{name_a} vs {name_b}",
            "experiment_a": name_a,
            "experiment_b": name_b,
            "n_common": result["n_common"],
            "accuracy_a": result["acc_a"],
            "accuracy_b": result["acc_b"],
            "diff_b_minus_a": result["diff"],
            "diff_b_minus_a_pp": result["diff_pp"],
            "ci_lower_pp": ci_lower * 100,
            "ci_upper_pp": ci_upper * 100,
            "both_right": result["both_right"],
            "a_only": result["a_only"],
            "b_only": result["b_only"],
            "both_wrong": result["both_wrong"],
            "discordant": result["discordant"],
            "p_value": result["p_value"],
        })

    if not rows:
        print("No comparisons passed the min-common threshold.")
        return

    result_df = pd.DataFrame(rows)

    # Multiple-comparison corrections.
    p_values = result_df["p_value"].tolist()
    result_df["p_bonferroni"] = bonferroni(p_values)
    result_df["p_bh_fdr"] = benjamini_hochberg(p_values)

    result_df["sig_uncorrected"] = result_df["p_value"].apply(significance)
    result_df["sig_bonferroni"] = result_df["p_bonferroni"].apply(significance)
    result_df["sig_bh_fdr"] = result_df["p_bh_fdr"].apply(significance)

    # Sort by p-value first, then absolute effect size.
    result_df = result_df.sort_values(
        by=["p_value", "diff_b_minus_a_pp"],
        ascending=[True, False],
    )

    out_csv = Path(f"{args.output_prefix}.csv")
    out_txt = Path(f"{args.output_prefix}.txt")
    metadata_csv = Path(f"{args.output_prefix}_loaded_experiments.csv")

    result_df.to_csv(out_csv, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata_csv, index=False)

    lines = []
    lines.append("=" * 70)
    lines.append("ALL-PAIRS MCNEMAR RESULTS")
    lines.append("=" * 70)
    lines.append(f"Experiments loaded: {len(all_results)}")
    lines.append(f"Comparisons run: {len(result_df)}")
    lines.append(f"Minimum common cases: {args.min_common}")
    lines.append("")
    lines.append("Top comparisons by unadjusted p-value:")
    lines.append("")

    header = (
        f"{'Comparison':<70} {'N':>5} {'A acc':>8} {'B acc':>8} "
        f"{'Diff':>9} {'p':>9} {'BH/FDR':>9} {'95% CI':>22}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in result_df.head(40).iterrows():
        comparison = row["comparison"][:69]
        line = (
            f"{comparison:<70} "
            f"{int(row['n_common']):>5} "
            f"{row['accuracy_a'] * 100:>7.1f}% "
            f"{row['accuracy_b'] * 100:>7.1f}% "
            f"{row['diff_b_minus_a_pp']:>+8.1f}pp "
            f"{row['p_value']:>9.4f} "
            f"{row['p_bh_fdr']:>9.4f} "
            f"[{row['ci_lower_pp']:>+6.1f}, {row['ci_upper_pp']:>+6.1f}]pp"
        )
        lines.append(line)

    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- Diff is experiment_b accuracy minus experiment_a accuracy.")
    lines.append("- Positive diff means experiment_b performed better on the common cases.")
    lines.append("- McNemar p-values use only discordant pairs: A-only vs B-only.")
    lines.append("- p_value is unadjusted; p_bonferroni and p_bh_fdr are adjusted for all pairwise tests in this run.")
    lines.append("- For thesis wording, report whether p-values are uncorrected or corrected.")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines[:50]))

    print(f"\nSaved full results to: {out_csv}")
    print(f"Saved text summary to: {out_txt}")
    print(f"Saved loading metadata to: {metadata_csv}")


if __name__ == "__main__":
    main()
