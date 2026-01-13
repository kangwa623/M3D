from datasets import load_dataset
import os
import json

BASE_DIR = os.path.join("datasets", "ct-rate-mini", "data")
IMG_DIR = os.path.join(BASE_DIR, "images")
NUM_SAMPLES = 15

os.makedirs(IMG_DIR, exist_ok=True)

print("Streaming CT-RATE (reports config) and downloading a mini subset...")

# NOTE: specify the "reports" config
ds = load_dataset(
    "ibrahimhamamci/CT-RATE",
    "reports",
    split="train",
    streaming=True
)

records = []

for i, sample in enumerate(ds):
    if i >= NUM_SAMPLES:
        break

    image = sample["image"]
    report = sample["report"]

    img_name = f"case_{i:03d}.nii.gz"
    img_path = os.path.join(IMG_DIR, img_name)

    image.save(img_path)

    records.append({
        "image": f"images/{img_name}",
        "text": report
    })

ann_path = os.path.join(BASE_DIR, "annotations.json")
with open(ann_path, "w") as f:
    json.dump(records, f, indent=2)

print(f"Done. Saved {NUM_SAMPLES} CT-RATE samples to:")
print(f"  {BASE_DIR}")
print(f"  {ann_path}")