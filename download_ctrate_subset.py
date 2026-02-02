import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- CONFIGURATION ---
N_SAMPLES = 150
SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("Loading CT-RATE reports (Streaming mode)...")
# Automatically uses your local logged-in token
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading {N_SAMPLES} samples...")

for count, item in enumerate(tqdm(ds, total=N_SAMPLES)):
    if count >= N_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]
        report_text = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"

        # 1. Save the Report
        txt_path = os.path.join(SAVE_TXT_DIR, volume_name.replace(".nii.gz", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 2. Securely Download the Volume
        # This function looks at your CLI login and handles the '401' automatically
        downloaded_file = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=f"dataset/train/{volume_name}",
            repo_type="dataset"
        )
        
        # Move it from the cache to your local folder
        shutil.copy(downloaded_file, os.path.join(SAVE_IMG_DIR, volume_name))

    except Exception as e:
        print(f"\n[!] Error on sample {count}: {e}")

print(f"\nSuccess! 150 samples saved to {SAVE_IMG_DIR}")