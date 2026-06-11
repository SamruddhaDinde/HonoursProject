"""
Runner for the option-ranking fusion experiment.

Usage:
    python -m langgraph_exp.run_option_ranking --n 50
    python -m langgraph_exp.run_option_ranking --n 689

Optional resume/slicing:
    python -m langgraph_exp.run_option_ranking --n 689 --start 622

Research question:
    Does structured option-level evidence aggregation improve diagnostic accuracy
    compared with free-form fusion or forced/conditional debate?
"""

import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

import wandb

from .lg_loader import load_cases_for_graph
from .option_rank_graph import build_option_rank_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["all", "test", "train"], default="all")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=int, default=0, help="Start index for resume/slicing")
    return p.parse_args()


def _json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(obj)


def main():
    args = parse_args()
    cases = load_cases_for_graph(split=args.split, seed=args.seed, n=args.n)
    if args.start:
        cases_to_run = cases[args.start:]
    else:
        cases_to_run = cases

    graph = build_option_rank_graph()

    wandb.init(
        project="medical-multiagent",
        name=f"lg_option_ranking_{args.split}_n{len(cases)}_start{args.start}",
        config={
            "framework": "langgraph",
            "architecture": "option_ranking_fusion",
            "distributed_policy": "text=vignette+options_no_image; vision=image+options_no_vignette; fusion=no_raw_image",
            "vision_model": os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b"),
            "text_model": os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b"),
            "orch_model": os.getenv("LG_ORCH_MODEL", "medgemma1.5:4b"),
            "split": args.split,
            "seed": args.seed,
            "n_total_requested": len(cases),
            "start": args.start,
            "n_run": len(cases_to_run),
        },
    )

    table = wandb.Table(columns=[
        "index", "image_id", "ground_truth",
        "text_top", "vision_top", "final_answer",
        "text_correct", "vision_correct", "final_correct",
        "agents_agreed", "fusion_changed_from_text", "fusion_change_helped",
        "text_confidence", "vision_confidence",
        "text_scores", "vision_scores", "vision_support",
        "text_parse_failed", "vision_parse_failed",
        "key_visual_findings", "text_reasoning", "vision_reasoning", "final_output",
        "running_accuracy",
    ])

    correct = total = 0
    text_correct = vision_correct = 0
    fusion_changed = fusion_helped = fusion_hurt = 0
    agreed = 0

    for local_i, ex in enumerate(cases_to_run):
        i = args.start + local_i
        out = graph.invoke(dict(ex))

        gt = ex["ground_truth"]
        text_top = out.get("text_top_answer", "UNKNOWN")
        vision_top = out.get("vision_top_answer", "UNKNOWN")
        final = out.get("final_answer", "UNKNOWN")

        total += 1
        final_ok = final == gt
        text_ok = text_top == gt
        vision_ok = vision_top == gt

        if final_ok:
            correct += 1
        if text_ok:
            text_correct += 1
        if vision_ok:
            vision_correct += 1
        if text_top == vision_top and text_top != "UNKNOWN":
            agreed += 1

        changed = bool(out.get("fusion_changed_from_text", False))
        change_help = ""
        if changed:
            fusion_changed += 1
            if final_ok and not text_ok:
                change_help = "helped"
                fusion_helped += 1
            elif (not final_ok) and text_ok:
                change_help = "hurt"
                fusion_hurt += 1
            else:
                change_help = "neutral"

        acc = correct / total if total else 0.0

        table.add_data(
            i, ex["image_id"], gt,
            text_top, vision_top, final,
            "correct" if text_ok else "wrong",
            "correct" if vision_ok else "wrong",
            "correct" if final_ok else "wrong",
            "yes" if text_top == vision_top and text_top != "UNKNOWN" else "no",
            "yes" if changed else "no",
            change_help,
            out.get("text_confidence", 0), out.get("vision_confidence", 0),
            _json(out.get("text_scores", {})),
            _json(out.get("vision_scores", {})),
            _json(out.get("vision_support", {})),
            "yes" if out.get("text_parse_failed") else "no",
            "yes" if out.get("vision_parse_failed") else "no",
            out.get("key_visual_findings", "")[:1000],
            out.get("text_reasoning", "")[:1000],
            out.get("vision_reasoning", "")[:1000],
            out.get("final_output", "")[:1500],
            round(acc, 4),
        )

        wandb.log({
            "running/accuracy": acc,
            "running/text_accuracy": text_correct / total,
            "running/vision_accuracy": vision_correct / total,
        })

        print(
            f"[{i+1}/{len(cases)}] {ex['image_id']:04d} "
            f"text={text_top} vision={vision_top} final={final} gt={gt} "
            f"{'ok' if final_ok else 'x'}"
            f"{' CHG:'+change_help if changed else ''}"
        )

    acc = correct / total if total else 0.0
    tacc = text_correct / total if total else 0.0
    vacc = vision_correct / total if total else 0.0

    wandb.log({"results_table": table})
    wandb.summary["final/accuracy"] = acc
    wandb.summary["final/total"] = total
    wandb.summary["final/text_accuracy"] = tacc
    wandb.summary["final/vision_accuracy"] = vacc
    wandb.summary["final/agents_agreed"] = agreed
    wandb.summary["final/fusion_changed_from_text"] = fusion_changed
    wandb.summary["final/fusion_changes_helped"] = fusion_helped
    wandb.summary["final/fusion_changes_hurt"] = fusion_hurt

    print(f"\nFinal accuracy: {acc:.2%} ({correct}/{total})")
    print(f"Text top accuracy: {tacc:.2%} ({text_correct}/{total})")
    print(f"Vision top accuracy: {vacc:.2%} ({vision_correct}/{total})")
    print(f"Agents agreed: {agreed}/{total}")
    print(
        f"Fusion changed text answer: {fusion_changed}/{total} "
        f"(helped {fusion_helped}, hurt {fusion_hurt})"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
