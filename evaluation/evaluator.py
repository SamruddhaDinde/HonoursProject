"""
Evaluation and W&B logging for all experimental conditions.

Supports three logging shapes:
  - log_multi_agent()    : output-only, CoT, structured JSON (text + vision + meta)
  - log_single_agent()   : single-agent baselines (one model, one call)
  - (ThoughtComm has its own logging in main_thoughtcomm.py)

Design principle: every multi-agent condition logs the SAME columns so that
W&B tables are directly comparable and joinable on image_id for paired tests.
Individual agent answers are extracted and scored independently — not just the
meta agent's final answer — because disagreement analysis (§7.4) requires it.
"""

import wandb
import re


#  Answer extraction 

def extract_answer(response_text: str) -> str:
    """Extract the multiple-choice letter from an agent response.

    Returns 'UNKNOWN' if no letter A-E can be found. The caller should
    track how often this happens — a high UNKNOWN rate signals a prompt
    problem, not a model problem.
    """
    if not response_text:
        return "UNKNOWN"

    # Primary: 'ANSWER: X' pattern 
    match = re.search(r"ANSWER:\s*([A-E])", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    fallback = re.search(r"\b([A-E])\b", response_text[:200])
    if fallback:
        return fallback.group(1).upper()

    return "UNKNOWN"


def is_correct(predicted: str, ground_truth: str) -> bool:
    """Strict letter match for multiple choice."""
    return predicted.strip().upper() == ground_truth.strip().upper()


#  Evaluator class 

class Evaluator:
    """Logs experimental runs to W&B with comprehensive per-case detail.

    Multi-agent logging includes individual agent answers and correctness,
    agreement tracking, and format failure detection — aligned with
    ThoughtComm's logging schema for cross-condition analysis.
    """

    def __init__(self, run_name: str, config: dict):
        self.run = wandb.init(
            project="medical-multiagent",
            name=run_name,
            config=config,
        )

        # Meta-agent / system-level counters
        self.meta_correct = 0
        self.total = 0
        self.format_failures = 0

        # Individual agent counters (multi-agent only)
        self.text_correct = 0
        self.vision_correct = 0
        self.agreement_count = 0

        # Disagreement-conditioned counters
        self.disagree_total = 0
        self.disagree_meta_correct = 0
        self.agree_total = 0
        self.agree_meta_correct = 0

        self.condition_kind = None
        self.table = None

    #  Multi-agent logging 

    def log_multi_agent(
        self,
        image_id: int,
        question: str,
        ground_truth: str,
        text_agent_output: str,
        vision_agent_output: str,
        meta_output: str,
        brier_score: float = 0.0,
    ):
        """Log one case from a multi-agent run (output-only, CoT, or structured).

        Extracts and scores each agent's answer independently, tracks
        agreement, and logs everything needed for §7.4 disagreement analysis.
        """
        if self.table is None:
            self.condition_kind = "multi_agent"
            self.table = wandb.Table(columns=[
                "image_id",
                "question",
                "ground_truth",
                "brier_score",

                # Individual agent answers and correctness
                "text_predicted",
                "text_correct",
                "vision_predicted",
                "vision_correct",

                # Agreement
                "agents_agree",

                # Meta agent (system output)
                "meta_predicted",
                "meta_correct",

                # Format tracking
                "text_format_ok",
                "vision_format_ok",
                "meta_format_ok",

                # Full outputs for qualitative analysis
                "text_agent_output",
                "vision_agent_output",
                "meta_output",

                # Running metrics
                "running_meta_accuracy",
                "running_text_accuracy",
                "running_vision_accuracy",
                "running_agreement_rate",
            ])

        # Extract answers
        text_pred = extract_answer(text_agent_output)
        vision_pred = extract_answer(vision_agent_output)
        meta_pred = extract_answer(meta_output)

        # Score
        text_ok = is_correct(text_pred, ground_truth)
        vision_ok = is_correct(vision_pred, ground_truth)
        meta_ok = is_correct(meta_pred, ground_truth)
        agents_agree = (text_pred == vision_pred)

        # Format tracking
        text_fmt = (text_pred != "UNKNOWN")
        vision_fmt = (vision_pred != "UNKNOWN")
        meta_fmt = (meta_pred != "UNKNOWN")
        if not meta_fmt:
            self.format_failures += 1

        # Tally
        self.total += 1
        if text_ok: self.text_correct += 1
        if vision_ok: self.vision_correct += 1
        if meta_ok: self.meta_correct += 1

        if agents_agree:
            self.agreement_count += 1
            self.agree_total += 1
            if meta_ok: self.agree_meta_correct += 1
        else:
            self.disagree_total += 1
            if meta_ok: self.disagree_meta_correct += 1

        # Running metrics
        run_meta_acc = self.meta_correct / self.total
        run_text_acc = self.text_correct / self.total
        run_vision_acc = self.vision_correct / self.total
        run_agree_rate = self.agreement_count / self.total

        # Table row
        self.table.add_data(
            image_id,
            question[:150] + "..." if len(question) > 150 else question,
            ground_truth,
            round(brier_score, 4),

            text_pred,
            "correct" if text_ok else "wrong",
            vision_pred,
            "correct" if vision_ok else "wrong",

            "yes" if agents_agree else "no",

            meta_pred,
            "correct" if meta_ok else "wrong",

            "ok" if text_fmt else "FAIL",
            "ok" if vision_fmt else "FAIL",
            "ok" if meta_fmt else "FAIL",

            text_agent_output,
            vision_agent_output,
            meta_output,

            round(run_meta_acc, 4),
            round(run_text_acc, 4),
            round(run_vision_acc, 4),
            round(run_agree_rate, 4),
        )

        # Live W&B charts
        wandb.log({
            "running/meta_accuracy": run_meta_acc,
            "running/text_accuracy": run_text_acc,
            "running/vision_accuracy": run_vision_acc,
            "running/agreement_rate": run_agree_rate,
        })

        print(f"  Text: {text_pred}({'ok' if text_ok else 'x'}) | "
              f"Vision: {vision_pred}({'ok' if vision_ok else 'x'}) | "
              f"{'AGREE' if agents_agree else 'DISAGREE'} | "
              f"Meta: {meta_pred}({'ok' if meta_ok else 'x'}) | "
              f"GT: {ground_truth}")

    #  Single-agent logging ─

    def log_single_agent(
        self,
        image_id: int,
        question: str,
        ground_truth: str,
        agent_output: str,
        brier_score: float = 0.0,
    ):
        """Log one case from a single-agent baseline run."""
        if self.table is None:
            self.condition_kind = "single_agent"
            self.table = wandb.Table(columns=[
                "image_id",
                "question",
                "ground_truth",
                "brier_score",
                "predicted",
                "correct",
                "format_ok",
                "agent_output",
                "running_accuracy",
            ])

        predicted = extract_answer(agent_output)
        correct = is_correct(predicted, ground_truth)
        fmt_ok = (predicted != "UNKNOWN")

        if not fmt_ok:
            self.format_failures += 1
        if correct:
            self.meta_correct += 1
        self.total += 1

        run_acc = self.meta_correct / self.total

        self.table.add_data(
            image_id,
            question[:150] + "..." if len(question) > 150 else question,
            ground_truth,
            round(brier_score, 4),
            predicted,
            "correct" if correct else "wrong",
            "ok" if fmt_ok else "FAIL",
            agent_output,
            round(run_acc, 4),
        )

        wandb.log({
            "running/accuracy": run_acc,
        })

        print(f"  Predicted: {predicted} | GT: {ground_truth} | "
              f"{'correct' if correct else 'wrong'}")

    #  Finish 

    def finish(self):
        """Write final metrics and close W&B run."""
        if self.total == 0:
            wandb.finish()
            return 0.0

        meta_acc = self.meta_correct / self.total
        fmt_rate = self.format_failures / self.total

        # Common summary
        wandb.log({"results_table": self.table})
        wandb.summary["final/accuracy"] = meta_acc
        wandb.summary["final/total_examples"] = self.total
        wandb.summary["final/correct"] = self.meta_correct
        wandb.summary["final/format_failures"] = self.format_failures
        wandb.summary["final/format_failure_rate"] = fmt_rate

        print(f"\n{'='*50}")
        print(f"Final Accuracy: {meta_acc:.2%} ({self.meta_correct}/{self.total})")

        # Multi-agent specific summary
        if self.condition_kind == "multi_agent":
            text_acc = self.text_correct / self.total
            vision_acc = self.vision_correct / self.total
            agree_rate = self.agreement_count / self.total

            wandb.summary["final/text_accuracy"] = text_acc
            wandb.summary["final/vision_accuracy"] = vision_acc
            wandb.summary["final/agreement_rate"] = agree_rate
            wandb.summary["final/agree_total"] = self.agree_total
            wandb.summary["final/disagree_total"] = self.disagree_total

            if self.agree_total > 0:
                agree_acc = self.agree_meta_correct / self.agree_total
                wandb.summary["final/meta_accuracy_when_agree"] = agree_acc
                print(f"Meta accuracy when agents agree:    "
                      f"{agree_acc:.2%} ({self.agree_meta_correct}/{self.agree_total})")

            if self.disagree_total > 0:
                disagree_acc = self.disagree_meta_correct / self.disagree_total
                wandb.summary["final/meta_accuracy_when_disagree"] = disagree_acc
                print(f"Meta accuracy when agents disagree: "
                      f"{disagree_acc:.2%} ({self.disagree_meta_correct}/{self.disagree_total})")

            print(f"Text agent accuracy:  {text_acc:.2%} ({self.text_correct}/{self.total})")
            print(f"Vision agent accuracy: {vision_acc:.2%} ({self.vision_correct}/{self.total})")
            print(f"Agreement rate:        {agree_rate:.2%} ({self.agreement_count}/{self.total})")

        if self.format_failures > 0:
            print(f"Format failures:       {self.format_failures} ({fmt_rate:.1%})")

        print(f"{'='*50}")
        wandb.finish()
        return meta_acc