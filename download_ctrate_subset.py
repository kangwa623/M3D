import os
import requests
from datasets import load_dataset
from tqdm import tqdm

N_SAMPLES = 150
SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("Loading CT-RATE reports config...")
# This config works (we verified it in your last CLI test)
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Processing first {N_SAMPLES} samples...")

for count, item in enumerate(tqdm(ds, total=N_SAMPLES)):
    if count >= N_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]  # e.g., "train_0.nii.gz"
        report_text = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"

        # 1. Save the Report
        txt_path = os.path.join(SAVE_TXT_DIR, volume_name.replace(".nii.gz", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 2. Download the Image File 
        # The CT-RATE files are hosted in the 'volumes' folder of the repo
        file_url = f"https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/resolve/main/dataset/train/{volume_name}"
        img_path = os.path.join(SAVE_IMG_DIR, volume_name)

        # Download via requests (better control for large medical files)
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(img_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"Could not download {volume_name}. Status: {response.status_code}")

    except Exception as e:
        print(f"Error on sample {count}: {e}")

print(f"\nSuccess! Saved {N_SAMPLES} files to {SAVE_IMG_DIR}")