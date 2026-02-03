import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- CONFIGURATION ---
N_SAMPLES = 150
SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

# Create local folders
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("Loading CT-RATE metadata...")
# Load the 'reports' config to get the mapping of VolumeNames
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading {N_SAMPLES} .nii.gz files...")

for count, item in enumerate(tqdm(ds, total=N_SAMPLES)):
    if count >= N_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]  # This is the 'train_X.nii.gz' name
        
        # 1. Save the Report (as a .txt file)
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        with open(os.path.join(SAVE_TXT_DIR, txt_filename), "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Download the actual .nii.gz Volume
        # We target 'train_fixed' for the best NIfTI compatibility
        cached_file = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=f"dataset/train_fixed/{volume_name}", 
            repo_type="dataset"
        )
        
        # 3. Copy the .nii.gz file to your folder
        shutil.copy(cached_file, os.path.join(SAVE_IMG_DIR, volume_name))

    except Exception as e:
        print(f"\n Error downloading {volume_name}: {e}")
        continue

print(f"\n Done! Check the '{SAVE_IMG_DIR}' folder for your 150 .nii.gz files.")
