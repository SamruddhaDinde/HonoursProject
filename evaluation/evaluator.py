import wandb
import re

def extract_answer(response_text: str) -> str:
    """Extract the answer after ANSWER: label."""
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|REASONING|$)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "UNKNOWN"

def is_correct(predicted: str, ground_truth: str) -> bool:
    """
    Flexible matching for open-ended medical answers.
    Handles cases like 'right side' matching 'right',
    or 'no abnormality seen' matching 'not seen here'.
    """
    predicted_clean = predicted.lower().strip()
    ground_truth_clean = ground_truth.lower().strip()

    # Exact match
    if predicted_clean == ground_truth_clean:
        return True

    # Ground truth contained in predicted
    if ground_truth_clean in predicted_clean:
        return True

    # Predicted contained in ground truth
    if predicted_clean in ground_truth_clean:
        return True

    # Token overlap — check if any meaningful word matches
    # Filter out common filler words
    stopwords = {"the", "a", "an", "is", "are", "of", "in", "on", "at", "to", "no", "not"}
    predicted_tokens = set(predicted_clean.split()) - stopwords
    ground_truth_tokens = set(ground_truth_clean.split()) - stopwords

    if predicted_tokens and ground_truth_tokens:
        overlap = predicted_tokens & ground_truth_tokens
        if overlap:
            return True

    return False

class Evaluator:
    def __init__(self, run_name: str, config: dict):
        self.run = wandb.init(
            project="medical-multiagent",
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