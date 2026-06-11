"""
Runner for the conditional-debate experiment.

Usage:
    python -m langgraph_exp.run_conditional_debate --n 50
    python -m langgraph_exp.run_conditional_debate --n 689
"""

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

import wandb
from .lg_loader import load_cases_for_graph
from .conditional_graph import build_conditional_debate_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["all", "test", "train"], default="all")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _helped_or_hurt(initial_answer: str, final_answer: str, ground_truth: str) -> str:
    if initial_answer == final_answer:
        return ""
    if initial_answer != ground_truth and final_answer == ground_truth:
        return "helped"
    if initial_answer == ground_truth and final_answer != ground_truth:
        return "hurt"
    return "neutral"


def main():
    args = parse_args()
    cases = load_cases_for_graph(split=args.split, seed=args.seed, n=args.n)
    graph = build_conditional_debate_graph()

    wandb.init(
        project="medical-multiagent",
        name=f"lg_conditional_debate_{args.split}_n{len(cases)}",
        config={
            "framework": "langgraph",
            "architecture": "conditional_debate",
            "vision_model": os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b"),
            "text_model": os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b"),
            "orch_model": os.getenv("LG_ORCH_MODEL", "qwen2.5vl:7b"),
            "max_rounds": int(os.getenv("LG_MAX_DEBATE_ROUNDS", "2")),
            "split": args.split,
            "seed": args.seed,
            "n": len(cases),
        },
    )

    table = wandb.Table(columns=[
        "image_id", "ground_truth",
        "text_initial", "text_final", "vision_answer", "final_answer",
        "text_confidence", "vision_confidence",
        "agents_agreed", "debate_triggered", "final_mode", "routing_reason",
        "text_initial_correct", "vision_correct", "final_correct",
        "text_changed", "change_helped", "n_questions_asked", "n_rounds",
        "visual_questions", "image_description", "vision_reasoning",
        "text_assessment", "final_output", "running_accuracy",
    ])

    correct = total = 0
    debate_count = direct_count = 0
    changed = changed_helped = changed_hurt = 0
    debate_correct = direct_correct = 0

    # for i, ex in enumerate(cases):
    #     out = graph.invoke(dict(ex))
    #     gt = ex["ground_truth"]
    for i, ex in enumerate(cases):
        try:
            out = graph.invoke(dict(ex))
        except Exception as e:
            print(f"[{i + 1}/{len(cases)}] {ex['image_id']:04d} SKIPPED: {e}")
            continue
        gt = ex["ground_truth"]

        t_init = out.get("text_answer_initial", "UNKNOWN")
        t_final = out.get("text_answer", "UNKNOWN")
        v_ans = out.get("vision_answer", "UNKNOWN")
        f_ans = out.get("final_answer", "UNKNOWN")

        debate_triggered = bool(out.get("debate_triggered", False))
        final_ok = (f_ans == gt)
        text_changed = (t_init != t_final)
        change_label = _helped_or_hurt(t_init, f_ans, gt)

        total += 1
        if final_ok:
            correct += 1

        if debate_triggered:
            debate_count += 1
            if final_ok:
                debate_correct += 1
        else:
            direct_count += 1
            if final_ok:
                direct_correct += 1

        if text_changed:
            changed += 1
            if change_label == "helped":
                changed_helped += 1
            elif change_label == "hurt":
                changed_hurt += 1

        all_qs = out.get("visual_questions", [])
        n_qs = sum(len(r) for r in all_qs)
        n_rounds = out.get("round", 0)

        table.add_data(
            ex["image_id"], gt,
            t_init, t_final, v_ans, f_ans,
            out.get("text_confidence", 1), out.get("vision_confidence", 1),
            "yes" if out.get("agents_agreed") else "no",
            "yes" if debate_triggered else "no",
            out.get("final_mode", ""), out.get("routing_reason", ""),
            "correct" if t_init == gt else "wrong",
            "correct" if v_ans == gt else "wrong",
            "correct" if final_ok else "wrong",
            "yes" if text_changed else "no",
            change_label,
            n_qs, n_rounds,
            " || ".join(" | ".join(r) for r in all_qs)[:1000],
            out.get("image_description", "")[:1200],
            out.get("vision_reasoning", "")[:1000],
            out.get("text_assessment", "")[:1000],
            out.get("final_output", "")[:1500],
            round(correct / total, 4),
        )

        wandb.log({"running/accuracy": correct / total})
        print(
            f"[{i + 1}/{len(cases)}] {ex['image_id']:04d} "
            f"text={t_init}->{t_final} vision={v_ans} final={f_ans} gt={gt} "
            f"{'DEBATE' if debate_triggered else 'DIRECT'} "
            f"{'ok' if final_ok else 'x'}"
        )

    acc = correct / total if total else 0.0
    direct_acc = direct_correct / direct_count if direct_count else 0.0
    debate_acc = debate_correct / debate_count if debate_count else 0.0

    wandb.log({"results_table": table})
    wandb.summary["final/accuracy"] = acc
    wandb.summary["final/total"] = total
    wandb.summary["routing/debate_count"] = debate_count
    wandb.summary["routing/direct_count"] = direct_count
    wandb.summary["routing/debate_accuracy"] = debate_acc
    wandb.summary["routing/direct_accuracy"] = direct_acc
    wandb.summary["final/text_changed_count"] = changed
    wandb.summary["final/changes_helped"] = changed_helped
    wandb.summary["final/changes_hurt"] = changed_hurt

    print(f"\nFinal accuracy: {acc:.2%} ({correct}/{total})")
    print(f"Direct decisions: {direct_count}/{total}, accuracy={direct_acc:.2%}")
    print(f"Debated decisions: {debate_count}/{total}, accuracy={debate_acc:.2%}")
    print(f"Text changed during debate: {changed}/{total} "
          f"(helped {changed_helped}, hurt {changed_hurt})")

    wandb.finish()


if __name__ == "__main__":
    main()
