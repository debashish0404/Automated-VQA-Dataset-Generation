"""
Image Download Utility

This script downloads images from a JSONL file containing image URLs.
Downloaded images are stored locally and validated using a minimum file size threshold.

Features:
- Reads image URLs from JSONL
- Downloads images sequentially
- Skips already downloaded files
- Basic validation using file size
- Retry-safe workflow
"""
from pathlib import Path
import json
import requests
import time
import random

# ================= FOLDER =================
PROJECT_FOLDER = Path("./data")
JSONL_PATH = PROJECT_FOLDER / "edu&occ.jsonl"
OUTPUT_DIR = PROJECT_FOLDER / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= SETTINGS =================
MAX_IMAGES = 100
MIN_SIZE = 10000  # 10KB filter

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("🚀 Starting image download...\n")

# ================= LOAD DATA =================
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

data = data[:MAX_IMAGES]

success = 0
failed = 0

# ================= DOWNLOAD LOOP =================
for item in data:
    img_id = item.get("id")
    url = item.get("image_url")

    if not url:
        print(f" No URL: {img_id}")
        failed += 1
        continue

    file_path = OUTPUT_DIR / f"{img_id}.jpg"

    # skip if already downloaded
    if file_path.exists() and file_path.stat().st_size > MIN_SIZE:
        print(f"⏩ Already exists: {img_id}")
        success += 1
        continue

    try:
        r = requests.get(url, headers=headers, timeout=20)

        if r.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(r.content)

            if file_path.stat().st_size > MIN_SIZE:
                print(f" Downloaded: {img_id}")
                success += 1
            else:
                file_path.unlink(missing_ok=True)
                failed += 1
        else:
            print(f" Failed HTTP {r.status_code}: {img_id}")
            failed += 1

    except Exception as e:
        print(f" Error {img_id}: {e}")
        failed += 1

    time.sleep(random.uniform(0.5, 1.5))  # anti-block

print("\n====================")
print(f" Success: {success}")
print(f" Failed : {failed}")
print(f" Saved  : {OUTPUT_DIR}")
