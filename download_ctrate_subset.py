import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

N_SAMPLES = 150
SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("Loading CT-RATE metadata...")
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train")

print(f"Downloading {N_SAMPLES} CT scans + reports...")

for i, item in enumerate(tqdm(ds)):
    if i >= N_SAMPLES:
        break

    volume_name = item["VolumeName"]      # e.g. train_123.nii.gz

    # ------------------ SAVE REPORT ------------------
    report = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
    with open(os.path.join(SAVE_TXT_DIR, volume_name.replace(".nii.gz", ".txt")), "w") as f:
        f.write(report)

    # ------------------ DOWNLOAD VOLUME ------------------
    try:
        file_path = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=f"dataset/{volume_name}",   # ✅ CORRECT PATH
            repo_type="dataset"
        )

        shutil.copy(file_path, os.path.join(SAVE_IMG_DIR, volume_name))

    except Exception as e:
        print(f"Skipping {volume_name}: {e}")

print("Done. 150 CT volumes saved.")
