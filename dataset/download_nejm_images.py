#!/usr/bin/env python3
"""
NEJM Image Challenge - Bulk Image Downloader

Reads the case JSON, fetches each image from NEJM's public image CDN,
and saves it under ./images/{image_id}.jpg.


Run:
    python download_nejm_images.py

Idempotent: rerun anytime to retry only the missing / failed images.
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


# Configuration 

SCRIPT_DIR = Path(__file__).parent.resolve()
JSON_PATH = SCRIPT_DIR / "image_challenge_dataset_20231223.json"
IMAGES_DIR = SCRIPT_DIR / "images"
FAILURE_LOG = SCRIPT_DIR / "download_failures.log"

URL_TEMPLATE = "https://csvc.nejm.org/ContentServer/images?id=IC{date}"

REQUEST_TIMEOUT = 30       
DELAY_BETWEEN_REQUESTS = 1.0  
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0      


HEADERS = {
    "User-Agent": (
        "Academic research script (Honours thesis, multimodal medical VQA). "
        "Contact: your_email@university.edu"
    ),
    "Accept": "image/jpeg,image/png,image/*,*/*",
}


# Helpers

def parse_date(date_str: str) -> str:
    """Convert 'apr-01-2021' -> '20210401'.

    NEJM URLs use YYYYMMDD; the JSON uses 'mon-dd-yyyy' lowercase.
    %b parses 3-letter month abbreviations. We .title() the input
    so 'apr' becomes 'Apr', which strptime accepts on every platform.
    """
    dt = datetime.strptime(date_str.title(), "%b-%d-%Y")
    return dt.strftime("%Y%m%d")


def download_one(url: str, dest: Path) -> tuple[bool, str]:
    """Download a single image with retries. Returns (success, reason).

    On success, writes the bytes to `dest` and verifies it opens as an image.
    On failure, leaves no partial file behind.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                ctype = resp.headers.get("Content-Type", "")
                if not ctype.startswith("image/"):
                    return False, f"Non-image Content-Type: {ctype!r}"

                dest.write_bytes(resp.content)

                # Validate: does this actually open as an image? Catches
                # truncated downloads and HTML error pages served as 200.
                try:
                    with Image.open(dest) as img:
                        img.verify()
                except Exception as e:
                    dest.unlink(missing_ok=True)
                    return False, f"Corrupt image data: {e}"

                return True, "ok"

            if resp.status_code == 429:
                # Rate limited — back off harder than usual
                time.sleep(RETRY_BACKOFF_BASE ** attempt * 2)
                continue

            if resp.status_code == 404:
                return False, "404 Not Found"

            # Other 4xx/5xx — retry with backoff
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            return False, f"HTTP {resp.status_code}"

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            return False, f"Network error: {e}"

    return False, "Max retries exceeded"


# Main 

def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"JSON not found at {JSON_PATH}. "
            "Place the dataset JSON next to this script."
        )

    IMAGES_DIR.mkdir(exist_ok=True)

    with open(JSON_PATH) as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} cases from {JSON_PATH.name}")
    print(f"Saving images to {IMAGES_DIR}\n")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bad_date": 0}
    failures: list[str] = []

    pbar = tqdm(cases, desc="Downloading", unit="img")
    for case in pbar:
        image_id = case["image_id"]
        date_str = case["date"]
        dest = IMAGES_DIR / f"{image_id:04d}.jpg"

        # Idempotency: if file exists and is non-empty, leave it alone
        if dest.exists() and dest.stat().st_size > 0:
            stats["skipped"] += 1
            continue

        try:
            yyyymmdd = parse_date(date_str)
        except ValueError as e:
            stats["bad_date"] += 1
            failures.append(f"id={image_id} date={date_str!r} ERROR=date parse: {e}")
            continue

        url = URL_TEMPLATE.format(date=yyyymmdd)
        success, reason = download_one(url, dest)

        if success:
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1
            failures.append(
                f"id={image_id} date={date_str} url={url} ERROR={reason}"
            )

        pbar.set_postfix(
            ok=stats["downloaded"], skip=stats["skipped"], fail=stats["failed"]
        )
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Write failure log only if there were failures
    if failures:
        with open(FAILURE_LOG, "w") as f:
            f.write(f"Run at {datetime.now().isoformat()}\n")
            f.write("=" * 70 + "\n")
            f.write("\n".join(failures) + "\n")
        print(f"\nFailure details written to: {FAILURE_LOG}")


    print("Summary")

    print(f"  Downloaded this run : {stats['downloaded']}")
    print(f"  Skipped (already on disk): {stats['skipped']}")
    print(f"  Failed              : {stats['failed']}")
    print(f"  Date parse errors   : {stats['bad_date']}")
    print(f"\n  Total cases         : {len(cases)}")
    print(f"  Images on disk      : {len(list(IMAGES_DIR.glob('*.jpg')))}")
    print(f"\n  Images directory    : {IMAGES_DIR}")
    if failures:
        print(f"  Failure log         : {FAILURE_LOG}")
        print("\n  Tip: rerun this script to retry only the failed cases.")


if __name__ == "__main__":
    main()