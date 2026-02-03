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
# Automatically uses your successful CLI login credentials
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
        # Note: Volumes in CT-RATE v2 are under 'dataset/train_fixed/' or 'dataset/train/'
        # Based on repo structure, we check the most common path:
        downloaded_file = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=f"dataset/train/{volume_name}",
            repo_type="dataset"
        )
        
        # Copy from HF cache to your local folder
        shutil.copy(downloaded_file, os.path.join(SAVE_IMG_DIR, volume_name))

    except Exception as e:
        print(f"\n[!] Error downloading {volume_name}: {e}")
        continue

print(f"\nDownload complete! Files saved in {SAVE_IMG_DIR}")