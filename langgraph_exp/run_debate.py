"""
Runner for the directed-debate experiment.

Logs everything needed to answer the REAL question this experiment poses:
does directed visual querying change the text agent's answer, and when it does,
is the change for the better? That's the `text_changed` / `change_helped`
analysis below — more important than the headline accuracy.

Usage:
    python -m langgraph_exp.run_debate --n 50      # pilot
    python -m langgraph_exp.run_debate --n 689     # full
"""

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

import wandb
from .lg_loader import load_cases_for_graph
from .debate_graph import build_debate_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["all", "test", "train"], default="all")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cases = load_cases_for_graph(split=args.split, seed=args.seed, n=args.n)
    graph = build_debate_graph()

    wandb.init(
        project="medical-multiagent",
        name=f"lg_debate_{args.split}_n{len(cases)}",
        config={
            "framework": "langgraph", "architecture": "directed_debate",
            "vision_model": os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b"),
            "text_model": os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b"),
            "orch_model": os.getenv("LG_ORCH_MODEL", "qwen2.5vl:7b"),
            "max_rounds": int(os.getenv("LG_MAX_DEBATE_ROUNDS", "2")),
            "split": args.split, "seed": args.seed, "n": len(cases),
        },
    )

    table = wandb.Table(columns=[
        "image_id", "ground_truth",
        "text_initial", "text_final", "final_answer",
        "text_initial_correct", "final_correct",
        "text_changed", "change_helped", "n_questions_asked", "n_rounds",
        "visual_questions", "text_assessment", "final_output",
        "running_accuracy",
    ])

    correct = total = 0
    changed = changed_helped = changed_hurt = 0
    for i, ex in enumerate(cases):
        out = graph.invoke(dict(ex))
        gt = ex["ground_truth"]

        t_init = out.get("text_answer_initial", "UNKNOWN")
        t_final = out.get("text_answer", "UNKNOWN")
        f_ans = out.get("final_answer", "UNKNOWN")

        init_ok = (t_init == gt)
        final_ok = (f_ans == gt)
        t_changed = (t_init != t_final)

        # Did the debate-driven change to the TEXT answer help or hurt?
        change_helped = ""
        if t_changed:
            changed += 1
            if (t_final == gt) and (t_init != gt):
                change_helped = "helped"; changed_helped += 1
            elif (t_final != gt) and (t_init == gt):
                change_helped = "hurt"; changed_hurt += 1
            else:
                change_helped = "neutral"

        total += 1
        if final_ok:
            correct += 1

        all_qs = out.get("visual_questions", [])
        n_qs = sum(len(r) for r in all_qs)
        n_rounds = out.get("round", 0)

        table.add_data(
            ex["image_id"], gt,
            t_init, t_final, f_ans,
            "correct" if init_ok else "wrong",
            "correct" if final_ok else "wrong",
            "yes" if t_changed else "no", change_helped, n_qs, n_rounds,
            " || ".join(" | ".join(r) for r in all_qs)[:1000],
            out.get("text_assessment", "")[:1000],
            out.get("final_output", "")[:1500],
            round(correct / total, 4),
        )
        wandb.log({"running/accuracy": correct / total})
        print(f"[{i+1}/{len(cases)}] {ex['image_id']:04d} "
              f"init={t_init} final={t_final} sys={f_ans} gt={gt} "
              f"{'CHG:'+change_helped if t_changed else ''}")

    acc = correct / total if total else 0.0
    wandb.log({"results_table": table})
    wandb.summary["final/accuracy"] = acc
    wandb.summary["final/total"] = total
    wandb.summary["final/text_changed_count"] = changed
    wandb.summary["final/changes_helped"] = changed_helped
    wandb.summary["final/changes_hurt"] = changed_hurt
    print(f"\nFinal accuracy: {acc:.2%} ({correct}/{total})")
    print(f"Text answer changed by debate: {changed}/{total}  "
          f"(helped {changed_helped}, hurt {changed_hurt})")
    wandb.finish()


if __name__ == "__main__":
    main()