from datasets import load_dataset
import os
import nibabel as nib
from tqdm import tqdm

N_SAMPLES = 150
SAVE_IMG_DIR = "ctrate_volumes"
SAVE_TXT_DIR = "ctrate_reports"

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)

print("Loading CT-RATE Volumes (Streaming mode)...")
# We load the 'Volumes' repo for images and 'reports' for text
ds_images = load_dataset("ibrahimhamamci/CT-RATE-Volumes", split="train", streaming=True)
ds_reports = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading first {N_SAMPLES} samples...")

# Zip them together to get both scan and report in one loop
for count, (img_item, txt_item) in enumerate(tqdm(zip(ds_images, ds_reports), total=N_SAMPLES)):
    if count >= N_SAMPLES:
        break
    try:
        # According to your dict_keys output:
        volume = img_item["scan"] 
        volume_name = txt_item["VolumeName"] # e.g., "train_0.nii.gz"
        
        # Combine different report sections into one text block
        full_report = f"FINDINGS:\n{txt_item['Findings_EN']}\n\nIMPRESSIONS:\n{txt_item['Impressions_EN']}"

        # Save volume 
        img_path = os.path.join(SAVE_IMG_DIR, volume_name)
        nib.save(volume, img_path)

        # Save report
        txt_path = os.path.join(SAVE_TXT_DIR, volume_name.replace(".nii.gz", ".txt"))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_report)

    except Exception as e:
        print(f"Error at index {count}: {e}")
        continue

print(f"\nFinished. Saved {N_SAMPLES} volumes + reports.")