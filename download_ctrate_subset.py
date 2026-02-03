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

print("🚀 Loading CT-RATE reports...")
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"📦 Starting download of {N_SAMPLES} samples...")

count = 0

for item in tqdm(ds, total=N_SAMPLES):
    if count >= N_SAMPLES:
        break

    volume_name = item["VolumeName"]
    report_text = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"

    try:
        # Save report
        txt_path = os.path.join(SAVE_TXT_DIR, volume_name.replace(".nii.gz", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Try BOTH possible dataset folders
        possible_paths = [
            f"dataset/train_fixed/{volume_name}",
            f"dataset/train/{volume_name}",
        ]

        downloaded_path = None
        for path in possible_paths:
            try:
                downloaded_path = hf_hub_download(
                    repo_id="ibrahimhamamci/CT-RATE",
                    filename=path,
                    repo_type="dataset"
                )
                break
            except:
                continue

        if downloaded_path is None:
            print(f"⚠️ File not found on HF: {volume_name}")
            continue

        final_img_path = os.path.join(SAVE_IMG_DIR, volume_name)
        shutil.copy(downloaded_path, final_img_path)

        count += 1

    except Exception as e:
        print(f"⚠️ Error processing {volume_name}: {e}")
        continue

print(f"\n✅ Finished! {count} volumes and reports saved successfully.")
