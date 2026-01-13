from datasets import load_dataset
import os
import json
import shutil

# TODO: Change this path to wherever you place the official CT-RATE images
OFFICIAL_IMG_ROOT = os.path.expanduser("~/CT-RATE/images")

BASE_DIR = os.path.join("datasets", "ct-rate-mini", "data")
IMG_DIR = os.path.join(BASE_DIR, "images")
NUM_SAMPLES = 15

os.makedirs(IMG_DIR, exist_ok=True)

print("Building CT-RATE mini dataset from official images...")

ds = load_dataset(
    "ibrahimhamamci/CT-RATE",
    "reports",
    split="train",
    streaming=True
)

records = []
count = 0

for sample in ds:
    if count >= NUM_SAMPLES:
        break

    vol = sample["VolumeName"]  # e.g. train_1a_1.nii.gz
    src = os.path.join(OFFICIAL_IMG_ROOT, vol)

    # Skip if this CT volume is not present on disk
    if not os.path.exists(src):
        continue

    text = (
        sample.get("Findings_EN", "") + "\n" +
        sample.get("Impressions_EN", "")
    ).strip()

    dst_name = f"case_{count:03d}.nii.gz"
    dst = os.path.join(IMG_DIR, dst_name)

    shutil.copyfile(src, dst)

    records.append({
        "image": f"images/{dst_name}",
        "text": text
    })

    count += 1

with open(os.path.join(BASE_DIR, "annotations.json"), "w") as f:
    json.dump(records, f, indent=2)

print(f"Built CT-RATE mini set with {count} samples.")