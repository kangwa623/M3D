
### Put this inside:


from datasets import load_dataset
import os
import nibabel as nib
from tqdm import tqdm

# how many samples you want
N_SAMPLES = 120   # you said ~100, so take extra in case some fail

SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("Loading CT-RATE dataset from HuggingFace...")
ds = load_dataset("StanfordAIMI/CT-RATE", split="train")

print(f"Dataset size: {len(ds)}")
print(f"Downloading first {N_SAMPLES} samples...")

count = 0

for item in tqdm(ds):
    try:
        volume = item["image"]        # 3D CT volume
        report = item["report"]       # ground truth text
        case_id = item["study_id"]    # unique id

        # Save volume as .nii.gz
        img_path = os.path.join(SAVE_IMG_DIR, f"{case_id}.nii.gz")
        nib.save(volume, img_path)

        # Save report text
        txt_path = os.path.join(SAVE_TXT_DIR, f"{case_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)

        count += 1
        if count >= N_SAMPLES:
            break

    except Exception as e:
        print("Skipping one sample:", e)

print(f"\nFinished. Saved {count} volumes + reports.")