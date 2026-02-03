import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- CONFIGURATION ---
N_SAMPLES = 150
SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

# Ensure directories exist
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("🚀 Loading CT-RATE reports...")
# 'reports' config contains the VolumeName mapping we need
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"📦 Starting download of {N_SAMPLES} samples...")

for count, item in enumerate(tqdm(ds, total=N_SAMPLES)):
    if count >= N_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]  # e.g., 'train_0.nii.gz'
        report_text = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"

        # 1. Save the Report Text
        txt_path = os.path.join(SAVE_TXT_DIR, volume_name.replace(".nii.gz", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 2. Download the 3D Scan (Using the v2 fixed path)
        # hf_hub_download automatically uses your 'Success' CLI login
        downloaded_path = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=f"dataset/train_fixed/{volume_name}", 
            repo_type="dataset"
        )
        
        # Move file from the HF cache to your desired local folder
        final_img_path = os.path.join(SAVE_IMG_DIR, volume_name)
        shutil.copy(downloaded_path, final_img_path)

    except Exception as e:
        print(f"\n⚠️ Error on {volume_name}: {e}")
        continue

print(f"\n✅ Finished! {count} volumes and reports saved successfully.")