"""
Runner for the conservative conditional-debate experiment.

Usage:
    python -m langgraph_exp.run_conservative_conditional_debate --n 50
    python -m langgraph_exp.run_conservative_conditional_debate --n 689

Resume/subset:
    python -m langgraph_exp.run_conservative_conditional_debate --n 689 --start 622
"""

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

import wandb
from .lg_loader import load_cases_for_graph
from .conservative_graph import build_conservative_debate_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["all", "test", "train"], default="all")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
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

    if args.start < 0 or args.start >= len(cases):
        raise ValueError(f"--start must be between 0 and {len(cases) - 1}; got {args.start}")

    graph = build_conservative_debate_graph()
    run_cases = cases[args.start:]

    wandb.init(
        project="medical-multiagent",
        name=f"lg_conservative_conditional_debate_{args.split}_n{len(cases)}_start{args.start}",
        config={
            "framework": "langgraph",
            "architecture": "conservative_conditional_debate",
            "vision_model": os.getenv("LG_VISION_MODEL", "qwen2.5vl:7b"),
            "text_model": os.getenv("LG_TEXT_MODEL", "medgemma1.5:4b"),
            "max_rounds": int(os.getenv("LG_MAX_DEBATE_ROUNDS", "2")),
            "split": args.split,
            "seed": args.seed,
            "n_loaded": len(cases),
            "start": args.start,
            "n_run": len(run_cases),
            "final_decision": "deterministic_guarded_text_answer",
        },
    )

    table = wandb.Table(columns=[
        "case_index", "image_id", "ground_truth",
        "text_initial", "text_final", "proposed_text_answer", "final_answer",
        "text_initial_confidence", "text_final_confidence", "proposed_text_confidence",
        "image_diagnosticity", "debate_triggered", "final_mode", "routing_reason",
        "revision_allowed", "guard_blocked_change", "revision_guard_reason",
        "visual_evidence_strength", "visual_contradicts_previous", "visual_supports_new_answer",
        "text_initial_correct", "final_correct", "text_changed", "change_helped",
        "proposed_change", "approved_change", "n_questions_asked", "n_rounds",
        "visual_questions", "visual_answers", "image_description", "image_gate_reasoning",
        "text_initial_assessment", "text_assessment", "proposed_text_assessment",
        "final_output", "running_accuracy",
    ])

    correct = total = 0
    debate_count = direct_count = 0
    debate_correct = direct_correct = 0
    proposed_changes = approved_changes = blocked_changes = 0
    changed = changed_helped = changed_hurt = 0

    for local_i, ex in enumerate(run_cases):
        case_index = args.start + local_i
        out = graph.invoke(dict(ex))
        gt = ex["ground_truth"]

        t_init = out.get("text_answer_initial", "UNKNOWN")
        t_final = out.get("text_answer", "UNKNOWN")
        proposed = out.get("proposed_text_answer", "")
        f_ans = out.get("final_answer", "UNKNOWN")

        debate_triggered = bool(out.get("debate_triggered", False))
        final_ok = (f_ans == gt)
        text_changed = (t_init != t_final)
        proposed_change = bool(proposed and proposed != t_init and proposed != "UNKNOWN")
        approved_change = text_changed
        blocked_change = bool(out.get("guard_blocked_change", False))
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

        if proposed_change:
            proposed_changes += 1
        if approved_change:
            approved_changes += 1
        if blocked_change:
            blocked_changes += 1

        if text_changed:
            changed += 1
            if change_label == "helped":
                changed_helped += 1
            elif change_label == "hurt":
                changed_hurt += 1

        all_qs = out.get("visual_questions", [])
        all_as = out.get("visual_answers", [])
        n_qs = sum(len(r) for r in all_qs)
        n_rounds = out.get("round", 0)

        table.add_data(
            case_index, ex["image_id"], gt,
            t_init, t_final, proposed, f_ans,
            out.get("text_initial_confidence", 1), out.get("text_confidence", 1), int(out.get("proposed_text_confidence", 0) or 0),
            out.get("image_diagnosticity", 1),
            "yes" if debate_triggered else "no",
            out.get("final_mode", ""), out.get("routing_reason", ""),
            "yes" if out.get("revision_allowed") else "no",
            "yes" if blocked_change else "no",
            out.get("revision_guard_reason", ""),
            out.get("visual_evidence_strength", ""),
            "yes" if out.get("visual_contradicts_previous") else "no",
            "yes" if out.get("visual_supports_new_answer") else "no",
            "correct" if t_init == gt else "wrong",
            "correct" if final_ok else "wrong",
            "yes" if text_changed else "no",
            change_label,
            "yes" if proposed_change else "no",
            "yes" if approved_change else "no",
            n_qs, n_rounds,
            " || ".join(" | ".join(r) for r in all_qs)[:1200],
            " || ".join(" | ".join(r) for r in all_as)[:1200],
            out.get("image_description", "")[:1200],
            out.get("image_gate_reasoning", "")[:1000],
            out.get("text_initial_assessment", "")[:1000],
            out.get("text_assessment", "")[:1200],
            out.get("proposed_text_assessment", "")[:1000],
            out.get("final_output", "")[:1500],
            round(correct / total, 4),
        )

        wandb.log({"running/accuracy": correct / total})
        print(
            f"[{case_index + 1}/{len(cases)}] {ex['image_id']:04d} "
            f"text={t_init}->{t_final} proposed={proposed or '-'} final={f_ans} gt={gt} "
            f"{'DEBATE' if debate_triggered else 'DIRECT'} "
            f"{'BLOCK' if blocked_change else ''} "
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
    wandb.summary["revision/proposed_changes"] = proposed_changes
    wandb.summary["revision/approved_changes"] = approved_changes
    wandb.summary["revision/blocked_changes"] = blocked_changes
    wandb.summary["final/text_changed_count"] = changed
    wandb.summary["final/changes_helped"] = changed_helped
    wandb.summary["final/changes_hurt"] = changed_hurt

    print(f"\nFinal accuracy: {acc:.2%} ({correct}/{total})")
    print(f"Direct decisions: {direct_count}/{total}, accuracy={direct_acc:.2%}")
    print(f"Debated decisions: {debate_count}/{total}, accuracy={debate_acc:.2%}")
    print(
        f"Revision proposals: {proposed_changes}; approved: {approved_changes}; "
        f"blocked: {blocked_changes}"
    )
    print(
        f"Approved text changes: {changed}/{total} "
        f"(helped {changed_helped}, hurt {changed_hurt})"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
