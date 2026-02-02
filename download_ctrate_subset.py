from datasets import load_dataset
import os
import nibabel as nib
from tqdm import tqdm

N_SAMPLES = 150
SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("Loading CT-RATE dataset (Streaming mode)...")
# Added "default" config and streaming=True
ds = load_dataset("ibrahimhamamci/CT-RATE", "default", split="train", streaming=True)

print(f"Downloading first {N_SAMPLES} samples...")

count = 0
for item in tqdm(ds, total=N_SAMPLES):
    try:
        # Note: In streaming mode, item["scan"] is usually a path or a pre-loaded array
        volume = item["scan"] 
        report = item["report"]
        case_id = item["volume_nodesave"] # Or "study_id" depending on the config schema

        # Save volume 
        img_path = os.path.join(SAVE_IMG_DIR, f"{case_id}.nii.gz")
        
        # If 'volume' is already a nibabel object, use nib.save
        # If it's a numpy array, you'll need to wrap it: nib.Nifti1Image(volume, affine=np.eye(4))
        nib.save(volume, img_path)

        # Save report
        txt_path = os.path.join(SAVE_TXT_DIR, f"{case_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)

        count += 1
        if count >= N_SAMPLES:
            break

    except Exception as e:
        print(f"Skipping sample {count} due to error: {e}")
        continue

print(f"\nFinished. Saved {count} volumes + reports.")