"""
Statistical Analysis: McNemar's Paired Test + Bootstrap CIs

This script loads per-case results from W&B CSV exports and computes:
  1. Pairwise exact McNemar tests
  2. Bootstrap 95% confidence intervals for accuracy differences
  3. A thesis-ready summary table

Run from your project root:

    python data_analysis/analysis_mcnemar_fixed.py

Optional:

    python data_analysis/analysis_mcnemar_fixed.py --csv-dir wandb_exports --n-boot 10000

Fixes included:
  - Uses pathlib paths instead of Windows backslashes like "wandb_exports\\..."
  - Opens CSVs with encoding="utf-8-sig" to avoid Windows cp1252 UnicodeDecodeError
  - Implements exact McNemar using scipy.stats.binomtest
  - Robustly parses booleans: correct/incorrect, true/false, ok/x, yes/no, 1/0, ticks/crosses
  - Uses common image_id intersection for fair paired comparisons
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
from scipy.stats import binomtest


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

DEFAULT_CSV_DIR = Path("wandb_exports")

CSV_FILENAMES = {
    "mode1_cot": "mode1_cot_689.csv",
    "mode2_debate": "mode2_debate_689.csv",
    "mode3_structured": "mode3_structured_689.csv",
    "mode3b_structured_deb": "mode3b_structured_debate_689.csv",
    "lg_describe_fuse": "lg_describe_fuse_689.csv",
    "lg_directed_debate": "lg_directed_debate_689.csv",
    "lg_conditional": "lg_conditional_debate_689.csv",
    "lg_option_ranking": "lg_option_ranking_689.csv",
    "lg_conservative": "lg_conservative_689.csv",
    "rag_text": "rag_text_689.csv",
}

# Preferred column names for each condition.
# The loader also has fallback detection if the preferred column is missing.
COLUMN_MAP = {
    # OpenAI Agents SDK / earlier modes
    "mode1_cot": {"id": "image_id", "correct": "meta_correct"},
    "mode2_debate": {"id": "image_id", "correct": "meta_correct"},
    "mode3_structured": {"id": "image_id", "correct": "meta_correct"},
    "mode3b_structured_deb": {"id": "image_id", "correct": "meta_correct"},
    "rag_text": {"id": "image_id", "correct": "meta_correct"},

    # LangGraph experiments
    "lg_describe_fuse": {"id": "image_id", "correct": "final_correct"},
    "lg_directed_debate": {"id": "image_id", "correct": "final_correct"},
    "lg_conditional": {"id": "image_id", "correct": "final_correct"},
    "lg_option_ranking": {"id": "image_id", "correct": "final_correct"},
    "lg_conservative": {"id": "image_id", "correct": "final_correct"},
}

COMPARISONS = [
    # Communication mechanism comparisons
    ("mode1_cot", "mode2_debate", "Mode 1 (CoT) vs Mode 2 (Debate)"),
    ("mode1_cot", "mode3_structured", "Mode 1 (CoT) vs Mode 3 (Structured)"),
    ("mode2_debate", "mode3_structured", "Mode 2 (Debate) vs Mode 3 (Structured)"),
    ("mode2_debate", "mode3b_structured_deb", "Mode 2 (Debate) vs Mode 3b (Struct+Deb)"),

    # Architecture / LangGraph comparisons
    ("mode1_cot", "lg_conditional", "Mode 1 vs Conditional Debate (LG)"),
    ("mode2_debate", "lg_conditional", "Mode 2 vs Conditional Debate (LG)"),
    ("lg_describe_fuse", "lg_conditional", "Describe-fuse vs Conditional Debate"),
    ("lg_directed_debate", "lg_conditional", "Directed Debate vs Conditional Debate"),
    ("lg_conditional", "lg_option_ranking", "Conditional Debate vs Option Ranking"),
    ("lg_conditional", "lg_conservative", "Conditional vs Conservative Conditional"),

    # RAG comparison
    ("mode1_cot", "rag_text", "Mode 1 vs RAG-enhanced Mode 1"),
]


# ──────────────────────────────────────────────────────────────────────
# Robust parsing helpers
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
    """Convert common W&B/export correctness values into bool.

    Returns True, False, or None if the value cannot be interpreted.
    """
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

    # Handle strings like "Correct.", "wrong,", etc.
    s_clean = s.rstrip(".,;:! ")
    if s_clean in TRUE_VALUES:
        return True
    if s_clean in FALSE_VALUES:
        return False

    return None


def normalize_image_id(value) -> str:
    """Normalise image IDs so IDs match across CSVs."""
    s = str(value).strip()

    # Convert "123.0" to "123" if needed.
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def find_column(fieldnames: Iterable[str], preferred: str, fallbacks: Iterable[str]) -> Optional[str]:
    """Find a column case-insensitively, first trying preferred then fallbacks."""
    fieldnames = list(fieldnames or [])
    lower_to_original = {c.lower().strip(): c for c in fieldnames}

    for cand in [preferred, *fallbacks]:
        if cand is None:
            continue
        key = cand.lower().strip()
        if key in lower_to_original:
            return lower_to_original[key]

    return None


# ──────────────────────────────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────────────────────────────


def load_from_csv(csv_path: Path, condition: str) -> Dict[str, bool]:
    """Load per-case correctness from a CSV export.

    Returns dict mapping image_id -> bool.
    """
    col = COLUMN_MAP.get(condition, {})
    preferred_id_col = col.get("id", "image_id")
    preferred_correct_col = col.get("correct", "final_correct")

    id_fallbacks = [
        "image_id", "case_id", "id", "case", "question_id",
    ]

    correct_fallbacks = [
        "final_correct", "meta_correct", "correct",
        "is_correct", "answer_correct", "prediction_correct",
        "text_correct", "vision_correct",
    ]

    results: Dict[str, bool] = {}

    # utf-8-sig handles normal UTF-8 and CSVs with BOM.
    # errors="replace" prevents one odd character from crashing the whole analysis.
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        id_col = find_column(fieldnames, preferred_id_col, id_fallbacks)
        correct_col = find_column(fieldnames, preferred_correct_col, correct_fallbacks)

        if id_col is None:
            raise ValueError(
                f"Could not find an image_id column in {csv_path}. "
                f"Available columns: {fieldnames}"
            )

        if correct_col is None:
            raise ValueError(
                f"Could not find a correctness column in {csv_path}. "
                f"Tried preferred={preferred_correct_col!r} plus fallbacks={correct_fallbacks}. "
                f"Available columns: {fieldnames}"
            )

        skipped = 0
        for row in reader:
            img_raw = row.get(id_col)
            correct_raw = row.get(correct_col)

            img_id = normalize_image_id(img_raw)
            is_correct = parse_bool(correct_raw)

            if not img_id or is_correct is None:
                skipped += 1
                continue

            results[img_id] = is_correct

    print(f"    Using columns: id={id_col!r}, correct={correct_col!r}")
    if skipped:
        print(f"    Skipped {skipped} rows with missing/unparseable values")

    return results


def load_condition(condition: str, csv_dir: Path) -> Dict[str, bool]:
    filename = CSV_FILENAMES.get(condition)
    if not filename:
        print(f"  SKIP {condition}: no filename configured")
        return {}

    path = csv_dir / filename
    if not path.exists():
        print(f"  SKIP {condition}: CSV not found at {path}")
        return {}

    return load_from_csv(path, condition)


# ──────────────────────────────────────────────────────────────────────
# McNemar test and bootstrap CI
# ──────────────────────────────────────────────────────────────────────


def mcnemar_test(results_a: Dict[str, bool], results_b: Dict[str, bool],
                 name_a: str, name_b: str) -> dict:
    """Run exact McNemar's test on paired binary outcomes.

    Contingency table:

                    B correct   B wrong
        A correct      a           b
        A wrong        c           d

    McNemar's exact test uses only discordant pairs b and c.
    """
    common_ids = sorted(set(results_a.keys()) & set(results_b.keys()))
    n = len(common_ids)

    if n == 0:
        return {"error": "No common cases", "n_common": 0}

    a = b = c = d = 0

    for img_id in common_ids:
        a_ok = results_a[img_id]
        b_ok = results_b[img_id]

        if a_ok and b_ok:
            a += 1
        elif a_ok and not b_ok:
            b += 1  # A only
        elif not a_ok and b_ok:
            c += 1  # B only
        else:
            d += 1

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
        statistic = 0
    else:
        # Exact two-sided McNemar = two-sided binomial test on min(b, c)
        # under H0 that b and c are equally likely.
        test = binomtest(k=min(b, c), n=discordant, p=0.5, alternative="two-sided")
        p_value = test.pvalue
        statistic = min(b, c)

    acc_a = (a + b) / n
    acc_b = (a + c) / n
    diff = acc_b - acc_a

    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_common": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "diff": diff,
        "diff_pp": diff * 100,
        "both_right": a,
        "a_only": b,
        "b_only": c,
        "both_wrong": d,
        "discordant": discordant,
        "p_value": p_value,
        "statistic": statistic,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


def bootstrap_ci(results_a: Dict[str, bool], results_b: Dict[str, bool],
                 n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI for paired accuracy difference, calculated as B - A."""
    common_ids = sorted(set(results_a.keys()) & set(results_b.keys()))
    n = len(common_ids)

    if n == 0:
        return (np.nan, np.nan, np.nan)

    a_correct = np.array([int(results_a[i]) for i in common_ids])
    b_correct = np.array([int(results_b[i]) for i in common_ids])

    rng = np.random.default_rng(seed=seed)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = b_correct[idx].mean() - a_correct[idx].mean()

    alpha = 1.0 - ci
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    mean_diff = diffs.mean()

    return lower, upper, mean_diff


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────


def significance_label(p_value: Optional[float]) -> str:
    if p_value is None or np.isnan(p_value):
        return "N/A"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def format_comparison(result: dict, ci_lower: float, ci_upper: float, label: str,
                      cond_a: str, cond_b: str) -> List[str]:
    sig = significance_label(result.get("p_value"))

    p_text = (
        f"{result['p_value']:.4f}"
        if result.get("p_value") is not None and not np.isnan(result["p_value"])
        else "N/A"
    )

    return [
        f"\n{'─' * 70}",
        label,
        f"  N common cases: {result['n_common']}",
        f"  {cond_a}: {result['acc_a']:.1%}",
        f"  {cond_b}: {result['acc_b']:.1%}",
        f"  Difference: {result['diff_pp']:+.1f}pp ({cond_b} − {cond_a})",
        (
            f"  Discordant pairs: {result['discordant']} "
            f"({result['a_only']} {cond_a}-only, {result['b_only']} {cond_b}-only)"
        ),
        f"  McNemar exact p-value: {p_text}",
        f"  Significance: {sig}",
        f"  Bootstrap 95% CI: [{ci_lower * 100:+.1f}pp, {ci_upper * 100:+.1f}pp]",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run McNemar paired tests and bootstrap CIs on W&B CSV exports."
    )
    parser.add_argument(
        "--csv-dir",
        default=str(DEFAULT_CSV_DIR),
        help="Directory containing W&B CSV exports. Default: wandb_exports",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10000,
        help="Number of bootstrap samples. Default: 10000",
    )
    parser.add_argument(
        "--output",
        default="analysis_mcnemar_results.txt",
        help="Output text file. Default: analysis_mcnemar_results.txt",
    )

    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_path = Path(args.output)

    print("=" * 70)
    print("McNemar's Paired Test + Bootstrap 95% CIs")
    print("=" * 70)
    print(f"CSV directory: {csv_dir.resolve()}")
    print(f"Bootstrap samples: {args.n_boot}")

    needed_conditions = sorted(set(c for pair in COMPARISONS for c in pair[:2]))

    print("\nLoading per-case results...")
    all_results: Dict[str, Dict[str, bool]] = {}

    for condition in needed_conditions:
        print(f"  Loading {condition}...")
        try:
            data = load_condition(condition, csv_dir)
        except Exception as e:
            print(f"    FAILED: {e}")
            data = {}

        if data:
            all_results[condition] = data
            print(f"    Loaded {len(data)} cases")
        else:
            print("    FAILED or empty — skipping comparisons involving this condition")

    output_lines: List[str] = []
    output_lines.append("=" * 70)
    output_lines.append("PAIRWISE STATISTICAL COMPARISONS")
    output_lines.append("=" * 70)

    print(f"\nRunning {len(COMPARISONS)} comparisons...\n")

    summary_rows = []

    for cond_a, cond_b, label in COMPARISONS:
        if cond_a not in all_results or cond_b not in all_results:
            lines = [
                f"\n{'─' * 70}",
                label,
                "  SKIPPED: missing data",
            ]
            for line in lines:
                print(line)
                output_lines.append(line)
            continue

        result = mcnemar_test(all_results[cond_a], all_results[cond_b], cond_a, cond_b)
        ci_lower, ci_upper, _ = bootstrap_ci(
            all_results[cond_a],
            all_results[cond_b],
            n_boot=args.n_boot,
        )

        lines = format_comparison(result, ci_lower, ci_upper, label, cond_a, cond_b)

        for line in lines:
            print(line)
            output_lines.append(line)

        summary_rows.append({
            "label": label,
            "diff_pp": result["diff_pp"],
            "p_value": result["p_value"],
            "ci_lower_pp": ci_lower * 100,
            "ci_upper_pp": ci_upper * 100,
            "n_common": result["n_common"],
            "a_only": result["a_only"],
            "b_only": result["b_only"],
            "significance": significance_label(result["p_value"]),
        })

    output_lines.append(f"\n\n{'=' * 70}")
    output_lines.append("THESIS-READY SUMMARY")
    output_lines.append(f"{'=' * 70}")
    output_lines.append(
        f"{'Comparison':<45} {'N':>5} {'Diff':>9} {'p-value':>10} {'95% CI':>22} {'Sig':>6}"
    )
    output_lines.append("-" * 105)

    print("\n\n" + "=" * 70)
    print("THESIS-READY SUMMARY")
    print("=" * 70)
    print(f"{'Comparison':<45} {'N':>5} {'Diff':>9} {'p-value':>10} {'95% CI':>22} {'Sig':>6}")
    print("-" * 105)

    for row in summary_rows:
        p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "N/A"
        label = row["label"][:44]
        line = (
            f"{label:<45} "
            f"{row['n_common']:>5} "
            f"{row['diff_pp']:>+8.1f}pp "
            f"{p_str:>10} "
            f"[{row['ci_lower_pp']:>+6.1f}, {row['ci_upper_pp']:>+6.1f}]pp "
            f"{row['significance']:>6}"
        )
        print(line)
        output_lines.append(line)

    # Save text results.
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    # Save machine-readable summary CSV too.
    summary_csv = output_path.with_suffix(".csv")
    if summary_rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            fieldnames = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nResults saved to: {output_path}")
    if summary_rows:
        print(f"Summary CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
