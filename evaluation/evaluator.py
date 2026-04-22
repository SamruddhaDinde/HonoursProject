import wandb
import re

def extract_answer(response_text: str) -> str:
    """Pull YES or NO out of agent response."""
    match = re.search(r"ANSWER:\s*(YES|NO)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"

class Evaluator:
    def __init__(self, run_name: str, config: dict):
        self.run = wandb.init(
            project="medical-multiagent",
            name=run_name,
            config=config
        )
        self.correct = 0
        self.total = 0

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
        is_correct = predicted == ground_truth.upper()

        if is_correct:
            self.correct += 1
        self.total += 1

        wandb.log({
            "example_idx": example_idx,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": int(is_correct),
            "running_accuracy": self.correct / self.total,
            "text_agent_output": text_agent_output,
            "vision_agent_output": vision_agent_output,
            "meta_output": meta_output,
            "question": question,
        })

        print(f"  Predicted: {predicted} | Ground truth: {ground_truth} | {'✓' if is_correct else '✗'}")

    def finish(self):
        final_accuracy = self.correct / self.total if self.total > 0 else 0
        wandb.summary["final_accuracy"] = final_accuracy
        wandb.summary["total_examples"] = self.total
        wandb.summary["correct"] = self.correct
        print(f"\nFinal Accuracy: {final_accuracy:.2%} ({self.correct}/{self.total})")
        wandb.finish()
        return final_accuracy