from datasets import load_dataset
import os, json, shutil

OFFICIAL_IMG_ROOT = os.path.expanduser("~/datasets/ct-rate-images")
BASE_DIR = os.path.join("datasets", "ct-rate-mini", "data")
IMG_DIR = os.path.join(BASE_DIR, "images")
NUM_SAMPLES = 15

os.makedirs(IMG_DIR, exist_ok=True)

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

    vol = sample["VolumeName"]           # e.g. train_1a_1.nii.gz
    src = os.path.join(OFFICIAL_IMG_ROOT, vol)

    if not os.path.exists(src):
        continue  # skip if that CT is not in your archive

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