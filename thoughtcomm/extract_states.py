"""
PHASE 0: Extract Hidden States from MedGemma

For every NEJM case, run MedGemma twice:
  - As a VISION agent (image + diagnostic question, no patient history)
  - As a TEXT agent (full clinical context + question, no image)

Extract the hidden state vector from the last transformer layer's last
token position for each agent, and save everything to disk.

This is a preprocessing pass. It runs once and produces the data needed
for all subsequent training phases.

Run:
    python -m thoughtcomm.extract_states

Outputs:
    artifacts/hidden_states_train.pt   (350 cases)
    artifacts/hidden_states_test.pt    (337 cases)
    artifacts/model_config.pt          (model dimensions)
"""

import os
import sys
import json
import torch
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path so we can import the data loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import (
    load_nejm,
    format_text_question,
    format_vision_question,
)
from thoughtcomm.model_loader import load_medgemma, extract_hidden_state


#  Configuration 
ARTIFACTS_DIR = Path(os.getenv("THOUGHTCOMM_ARTIFACTS", "artifacts"))
TRAIN_SPLIT = int(os.getenv("THOUGHTCOMM_TRAIN_SPLIT", "350"))
SEED = 42  # MUST match your baseline runs for fair comparison


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    #  Load model ─
    print("=" * 60)
    print("PHASE 0: Hidden State Extraction")
    print("=" * 60)

    model, processor, model_config = load_medgemma()
    hidden_size = model_config["hidden_size"]
    print(f"Hidden size (d_model): {hidden_size}")
    print(f"Concatenated H_t size: {2 * hidden_size}")

    #  Load dataset ─
    print(f"\nLoading ALL NEJM cases (seed={SEED})...")
    dataset = load_nejm(n_samples=None, seed=SEED)
    print(f"Loaded {len(dataset)} cases.")

    if TRAIN_SPLIT >= len(dataset):
        raise ValueError(
            f"TRAIN_SPLIT={TRAIN_SPLIT} >= dataset size={len(dataset)}. "
            f"Reduce TRAIN_SPLIT or check dataset."
        )

    #  Extract hidden states 
    all_data = []
    start_time = time.time()

    for i, case in enumerate(dataset):
        case_start = time.time()

        #  Vision Agent: sees image + diagnostic question only 
        vision_prompt = format_vision_question(case)
        h_vision, vision_response = extract_hidden_state(
            model, processor,
            text=vision_prompt,
            image=case["image"],
        )

        #  Text Agent: sees full clinical context + question, no image ─
        text_prompt = format_text_question(case)
        h_text, text_response = extract_hidden_state(
            model, processor,
            text=text_prompt,
            image=None,  # Text agent does NOT see the image
        )

        #  Concatenate: H_t = [H_vision ; H_text] 
        h_t = torch.cat([h_vision, h_text], dim=0)  # [2 * hidden_size]

        # Sanity checks
        assert h_vision.shape == (hidden_size,), \
            f"Vision hidden state shape mismatch: {h_vision.shape}"
        assert h_text.shape == (hidden_size,), \
            f"Text hidden state shape mismatch: {h_text.shape}"
        assert h_t.shape == (2 * hidden_size,), \
            f"Concatenated H_t shape mismatch: {h_t.shape}"

        all_data.append({
            "case_id": case["image_id"],
            "H_t": h_t,                           # [2 * d_model]
            "H_vision": h_vision,                  # [d_model]
            "H_text": h_text,                      # [d_model]
            "vision_response": vision_response,
            "text_response": text_response,
            "correct_answer": case["answer"],
            "question": case["question"],
            "options": case["options"],
            "image_id": case["image_id"],
        })

        elapsed = time.time() - case_start
        total_elapsed = time.time() - start_time
        rate = (i + 1) / total_elapsed
        remaining = (len(dataset) - i - 1) / rate if rate > 0 else 0

        print(f"  [{i+1}/{len(dataset)}] Case {case['image_id']:04d} "
              f"({elapsed:.1f}s) | "
              f"ETA: {remaining/60:.0f}min | "
              f"H_t norm: {h_t.norm():.1f}")

    #  Split into train/test 
    train_data = all_data[:TRAIN_SPLIT]
    test_data = all_data[TRAIN_SPLIT:]

    print(f"\nSplit: {len(train_data)} train / {len(test_data)} test")

    #  Compute summary statistics ─
    train_norms = torch.stack([d["H_t"] for d in train_data]).norm(dim=1)
    test_norms = torch.stack([d["H_t"] for d in test_data]).norm(dim=1)

    print(f"\nHidden state statistics:")
    print(f"  Train H_t norm: mean={train_norms.mean():.1f}, "
          f"std={train_norms.std():.1f}, "
          f"min={train_norms.min():.1f}, max={train_norms.max():.1f}")
    print(f"  Test H_t norm:  mean={test_norms.mean():.1f}, "
          f"std={test_norms.std():.1f}, "
          f"min={test_norms.min():.1f}, max={test_norms.max():.1f}")

    # Check for degenerate states (all zeros = something broke)
    if train_norms.mean() < 1.0:
        print("\n⚠ WARNING: Hidden state norms are very small. "
              "This may indicate a problem with hidden state extraction.")

    #  Save to disk ─
    torch.save(train_data, ARTIFACTS_DIR / "hidden_states_train.pt")
    torch.save(test_data, ARTIFACTS_DIR / "hidden_states_test.pt")
    torch.save(model_config, ARTIFACTS_DIR / "model_config.pt")

    total_time = time.time() - start_time
    print(f"\n✓ Phase 0 complete in {total_time/60:.1f} minutes.")
    print(f"  Saved to: {ARTIFACTS_DIR}/")
    print(f"  - hidden_states_train.pt ({len(train_data)} cases)")
    print(f"  - hidden_states_test.pt  ({len(test_data)} cases)")
    print(f"  - model_config.pt        (hidden_size={hidden_size})")

    summary = {
        "hidden_size": hidden_size,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "total_cases": len(dataset),
        "seed": SEED,
        "model_id": model_config["model_id"],
        "train_norm_mean": float(train_norms.mean()),
        "extraction_time_minutes": total_time / 60,
    }
    with open(ARTIFACTS_DIR / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()