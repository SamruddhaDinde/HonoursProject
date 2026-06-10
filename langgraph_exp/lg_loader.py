"""
Loader adapter for the LangGraph experiments.

Deliberately thin: it REUSES your existing data/loader.py (same JSON, same
filtering, same seed/split logic) so the LangGraph runs evaluate on the exact
same 689 cases as the OpenAI Agents SDK runs. Do not re-implement loading here
or the two setups could silently diverge.

It only adds what the orchestrator graph needs that the old pipeline didn't:
the base64 image packed into the example dict up front.
"""

from data.loader import load_nejm, load_nejm_split, get_ground_truth, jpeg_file_to_base64, IMAGES_DIR


def load_cases_for_graph(split: str = "all", seed: int = 42, n: int | None = None):
    """Return a list of example dicts ready to seed OrchestratorState."""
    if split == "all":
        cases = load_nejm(n_samples=None, seed=seed)
    else:
        cases = load_nejm_split(split=split, seed=seed)

    if n is not None:
        cases = cases[:n]

    out = []
    for ex in cases:
        out.append({
            "image_id": ex["image_id"],
            "question": ex["question"],
            "options": ex["options"],
            "ground_truth": get_ground_truth(ex),
            "image_b64": jpeg_file_to_base64(str(IMAGES_DIR / f"{ex['image_id']:04d}.jpg")),
        })
    return out