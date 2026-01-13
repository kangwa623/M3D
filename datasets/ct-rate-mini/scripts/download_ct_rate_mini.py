from datasets import load_dataset
import os
import json

# Base directory for the mini CT-RATE dataset
BASE_DIR = os.path.join("datasets", "ct-rate-mini", "data")
IMG_DIR = os.path.join(BASE_DIR, "images")
NUM_SAMPLES = 15

os.makedirs(IMG_DIR, exist_ok=True)

print("Streaming CT-RATE and downloading a mini subset...")

# Stream CT-RATE without downloading everything
ds = load_dataset("ibrahimhamamci/CT-RATE", split="train", streaming=True)

records = []

for i, sample in enumerate(ds):
    if i >= NUM_SAMPLES:
        break

    # CT volume and report
    image = sample["image"]
    report = sample["report"]

    img_name = f"case_{i:03d}.nii.gz"
    img_path = os.path.join(IMG_DIR, img_name)

    # Save CT volume locally
    image.save(img_path)

    records.append({
        "image": f"images/{img_name}",
        "text": report
    })

# Write annotations.json
ann_path = os.path.join(BASE_DIR, "annotations.json")
with open(ann_path, "w") as f:
    json.dump(records, f, indent=2)

print(f"Done. Saved {NUM_SAMPLES} CT-RATE samples to:")
print(f"  {BASE_DIR}")
print(f"  {ann_path}")