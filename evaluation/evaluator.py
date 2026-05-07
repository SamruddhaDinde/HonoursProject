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
    def __init__(self, run_name: str, config: dict):
        self.run = wandb.init(
            project="medical-multiagent_nejm",
            name=run_name,
            config=config
        )
        self.correct = 0
        self.total = 0

        self.table = wandb.Table(columns=[
            "example_idx",
            "question",
            "ground_truth",
            "predicted",
            "correct",
            "text_agent_output",
            "vision_agent_output",
            "meta_output",
            "running_accuracy"
        ])

    def log_example(
        self,
        question: str,
        text_agent_output: str,
        vision_agent_output: str,
        meta_output: str,
        ground_truth: str,
        example_idx: int
    ):
        predicted = extract_answer(meta_output)
        correct = is_correct(predicted, ground_truth)

        if correct:
            self.correct += 1
        self.total += 1

        running_acc = self.correct / self.total

        self.table.add_data(
            example_idx,
            question,
            ground_truth,
            predicted,
            "✓" if correct else "✗",
            text_agent_output,
            vision_agent_output,
            meta_output,
            round(running_acc, 3)
        )

        wandb.log({
            "correct": int(correct),
            "running_accuracy": running_acc,
            "example_idx": example_idx
        })

        print(f"  Predicted: {predicted} | Ground truth: {ground_truth} | {'✓' if correct else '✗'}")

    def finish(self):
        final_accuracy = self.correct / self.total if self.total > 0 else 0

        wandb.log({"results_table": self.table})
        wandb.summary["final_accuracy"] = final_accuracy
        wandb.summary["total_examples"] = self.total
        wandb.summary["correct"] = self.correct

        print(f"\nFinal Accuracy: {final_accuracy:.2%} ({self.correct}/{self.total})")
        wandb.finish()
        return final_accuracy