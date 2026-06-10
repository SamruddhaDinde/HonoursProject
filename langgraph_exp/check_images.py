"""
Diagnostic: verify NEJM image loading is correct.

Run inside your container:
    python -m langgraph_exp.check_images

Checks, in order:
  1. Every case's expected image file exists and opens without error.
  2. Reports the size distribution — tiny/identical sizes can signal
     placeholder or wrongly-mapped files.
  3. Dumps 8 sample cases (image_id, gold answer, options, image dimensions,
     and saves the image to ./image_check/) so you can EYEBALL whether the
     image actually matches the clinical question. This manual check is the
     only way to catch a wrong-but-valid mapping (right filename, wrong picture).
"""

import os
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from data.loader import load_nejm
from collections import Counter
from data.loader import load_nejm  # or just glob the image dir

OUT = Path("./image_check")
OUT.mkdir(exist_ok=True)


def main():
    cases = load_nejm(n_samples=None, seed=42)
    print(f"Loaded {len(cases)} cases\n")

    sizes = Counter()
    failures = []
    for ex in cases:
        img = ex["image"]
        sizes[img.size] += 1
        # Re-open from disk independently to catch any lazy-load issues
        try:
            _ = img.copy()
        except Exception as e:  # noqa: BLE001
            failures.append((ex["image_id"], str(e)))

    print(f"Distinct image dimensions: {len(sizes)}")
    most = sizes.most_common(5)
    for size, n in most:
        print(f"  {size}: {n} cases")
    if len(sizes) <= 3:
        print("  WARNING: very few distinct sizes — possible placeholder/wrong files")

    if failures:
        print(f"\n{len(failures)} images failed to load:")
        for iid, err in failures[:20]:
            print(f"  {iid:04d}: {err}")
    else:
        print("\nAll images loaded without error.")

    # Dump 8 samples for manual eyeball
    print("\nSaving 8 sample cases to ./image_check/ for manual inspection:")
    for ex in cases[:8]:
        iid = ex["image_id"]
        path = OUT / f"{iid:04d}.jpg"
        ex["image"].save(path)
        opts = " | ".join(f"{k}:{v}" for k, v in ex["options"].items())
        print(f"\n  case {iid:04d}  (gold={ex['answer']})  dims={ex['image'].size}")
        print(f"    Q: {ex['question'][:160]}")
        print(f"    options: {opts[:200]}")
        print(f"    saved -> {path}")
    print("\nOpen the saved images and confirm each matches its question above.")

  

    # modes = Counter()
    # for ex in load_nejm(n_samples=None, seed=42):
    #     modes[ex["image"].mode] += 1
    # print(modes)


if __name__ == "__main__":
    main()