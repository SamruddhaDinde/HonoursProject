import wandb
import re


def extract_answer(response_text: str) -> str:
    """Extract the multiple-choice letter from the agent response."""
    # Primary: 'ANSWER: X' pattern (what we instructed the model to use)
    match = re.search(r"ANSWER:\s*([A-E])", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: any standalone A-E in the first 200 chars. Models occasionally
    # break format. We accept this rather than scoring as wrong, but log it.
    fallback = re.search(r"\b([A-E])\b", response_text[:200])
    if fallback:
        return fallback.group(1).upper()
    return "UNKNOWN"


def is_correct(predicted: str, ground_truth: str) -> bool:
    """Strict letter match for multiple choice."""
    return predicted.strip().upper() == ground_truth.strip().upper()


class Evaluator:
    """Logs experimental runs to W&B with per-case detail.

    Supports two logging shapes:
      - log_example()              : multi-agent runs (text + vision + meta)
      - log_single_agent_example() : single-agent baseline runs

    The W&B table schema differs between the two so each condition's table
    is self-contained — no empty columns to confuse you when you analyse.
    """

    def __init__(self, run_name: str, config: dict):
        self.run = wandb.init(
            project="medical-multiagent",
            name=run_name,
            config=config,
        )
        self.correct = 0
        self.total = 0
        self.format_failures = 0
        self.condition_kind = None  # set on first log call

        # Tables created lazily on first log call so we can pick the right
        # schema based on which log method is used.
        self.table = None

    # -- Multi-agent logging --------------------------------------------------

    def log_example(
        self,
        image_id: int,
        question: str,
        text_agent_output: str,
        vision_agent_output: str,
        meta_output: str,
        ground_truth: str,
    ):
        if self.table is None:
            self.condition_kind = "multi_agent"
            self.table = wandb.Table(columns=[
                "image_id",
                "question",
                "ground_truth",
                "predicted",
                "correct",
                "text_agent_output",
                "vision_agent_output",
                "meta_output",
                "running_accuracy",
            ])

        predicted = extract_answer(meta_output)
        self._tally(predicted, ground_truth)

        self.table.add_data(
            image_id,
            question,
            ground_truth,
            predicted,
            "correct" if is_correct(predicted, ground_truth) else "wrong",
            text_agent_output,
            vision_agent_output,
            meta_output,
            round(self.correct / self.total, 3),
        )

        wandb.log({
            "correct": int(is_correct(predicted, ground_truth)),
            "running_accuracy": self.correct / self.total,
        })
        print(f"  Predicted: {predicted} | Ground truth: {ground_truth} | "
              f"{'correct' if is_correct(predicted, ground_truth) else 'wrong'}")

    # -- Single-agent logging -------------------------------------------------

    def log_single_agent_example(
        self,
        image_id: int,
        question: str,
        agent_output: str,
        ground_truth: str,
    ):
        if self.table is None:
            self.condition_kind = "single_agent"
            self.table = wandb.Table(columns=[
                "image_id",
                "question",
                "ground_truth",
                "predicted",
                "correct",
                "agent_output",
                "running_accuracy",
            ])

        predicted = extract_answer(agent_output)
        self._tally(predicted, ground_truth)

        self.table.add_data(
            image_id,
            question,
            ground_truth,
            predicted,
            "correct" if is_correct(predicted, ground_truth) else "wrong",
            agent_output,
            round(self.correct / self.total, 3),
        )

        wandb.log({
            "correct": int(is_correct(predicted, ground_truth)),
            "running_accuracy": self.correct / self.total,
        })
        print(f"  Predicted: {predicted} | Ground truth: {ground_truth} | "
              f"{'correct' if is_correct(predicted, ground_truth) else 'wrong'}")

    # -- Internals ------------------------------------------------------------

    def _tally(self, predicted: str, ground_truth: str):
        if predicted == "UNKNOWN":
            self.format_failures += 1
        if is_correct(predicted, ground_truth):
            self.correct += 1
        self.total += 1

    def finish(self):
        final_accuracy = self.correct / self.total if self.total > 0 else 0
        format_failure_rate = self.format_failures / self.total if self.total > 0 else 0

        wandb.log({"results_table": self.table})
        wandb.summary["final_accuracy"] = final_accuracy
        wandb.summary["total_examples"] = self.total
        wandb.summary["correct"] = self.correct
        wandb.summary["format_failures"] = self.format_failures
        wandb.summary["format_failure_rate"] = format_failure_rate

        print(f"\nFinal Accuracy: {final_accuracy:.2%} ({self.correct}/{self.total})")
        if self.format_failures > 0:
            print(f"Format failures: {self.format_failures} ({format_failure_rate:.1%})")
        wandb.finish()
        return final_accuracy