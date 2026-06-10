"""
Entrypoint for the orchestrator LangGraph experiment.

Runs the graph over the NEJM cases, logs to W&B with a schema comparable to
your existing Evaluator, and records the search audit fields. LangSmith
tracing is automatic once the LANGSMITH_* env vars are set — no code here.

Usage:
    # paired pair — RUN BOTH for an interpretable result:
    python -m langgraph_exp.run_orchestrator --search on   --n 689
    python -m langgraph_exp.run_orchestrator --search off  --n 689

    # smoke test (do this first — 5 cases, ~minutes):
    python -m langgraph_exp.run_orchestrator --search off --n 5
"""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()

import wandb
from .lg_loader import load_cases_for_graph
from .graph import build_graph


def contamination_hit(snippets, ground_truth, options) -> bool:
    """Crude leak detector: did any snippet contain the gold option's text?
    Flags cases for manual review; not a perfect measure, but enough to
    report 'search leaked the answer in ~X% of cases'."""
    gold_text = options.get(ground_truth, "").lower().strip()
    if not gold_text:
        return False
    for s in snippets:
        if gold_text and gold_text in (s.get("content", "") + s.get("title", "")).lower():
            return True
    return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--search", choices=["on", "off"], default="off")
    p.add_argument("--split", choices=["all", "test", "train"], default="all")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    use_search = (args.search == "on")
    cases = load_cases_for_graph(split=args.split, seed=args.seed, n=args.n)
    graph = build_graph()

    run = wandb.init(
        project="medical-multiagent",
        name=f"lg_orchestrator_search-{args.search}_{args.split}_n{len(cases)}",
        config={
            "framework": "langgraph",
            "architecture": "orchestrator_describe_textsearch",
            "use_web_search": use_search,
            "vision_model": os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b"),
            "text_model": os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b"),
            "orch_model": os.getenv("LG_ORCH_MODEL", "medgemma1.5:4b"),
            "split": args.split, "seed": args.seed, "n": len(cases),
        },
    )

    table = wandb.Table(columns=[
        "image_id", "ground_truth", "final_answer", "final_correct",
        "text_answer", "text_correct",
        "search_used", "search_mentioned_answer", "n_queries",
        "image_description", "text_assessment", "final_output",
        "search_queries", "running_accuracy",
    ])

    correct = total = 0
    for i, ex in enumerate(cases):
        init = dict(ex)
        init["use_web_search"] = use_search
        out = graph.invoke(init)

        gt = ex["ground_truth"]
        fa = out.get("final_answer", "UNKNOWN")
        ta = out.get("text_answer", "UNKNOWN")
        snippets = out.get("search_snippets", [])
        # Audit only: recorded to the table column for later spot-checking,
        # never printed or summarised. Lets you defend the accuracy number
        # post-hoc without re-running.
        leaked = contamination_hit(snippets, gt, ex["options"]) if use_search else False

        total += 1
        if fa == gt:
            correct += 1

        table.add_data(
            ex["image_id"], gt, fa, "correct" if fa == gt else "wrong",
            ta, "correct" if ta == gt else "wrong",
            "yes" if use_search else "no",
            "yes" if leaked else "no",
            len(out.get("search_queries", [])),
            out.get("image_description", "")[:2000],
            out.get("text_assessment", "")[:2000],
            out.get("final_output", "")[:2000],
            " | ".join(out.get("search_queries", [])),
            round(correct / total, 4),
        )
        wandb.log({"running/accuracy": correct / total})
        print(f"[{i+1}/{len(cases)}] {ex['image_id']:04d} "
              f"final={fa}({'ok' if fa==gt else 'x'}) text={ta} gt={gt}")

    acc = correct / total if total else 0.0
    wandb.log({"results_table": table})
    wandb.summary["final/accuracy"] = acc
    wandb.summary["final/total"] = total
    print(f"\nFinal accuracy: {acc:.2%} ({correct}/{total})")
    wandb.finish()


if __name__ == "__main__":
    main()