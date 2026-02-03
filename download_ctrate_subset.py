import os
from datasets import load_dataset
from tqdm import tqdm

N_SAMPLES = 150
SAVE_DIR = "ctrate_train_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading CT-RATE training set...")
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train")

print(f"Saving first {N_SAMPLES} samples...")

for i in tqdm(range(N_SAMPLES)):
    item = ds[i]

    volume_name = item["VolumeName"]
    report_text = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"

    # Save report
    with open(os.path.join(SAVE_DIR, volume_name.replace(".nii.gz", ".txt")), "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save metadata (optional)
    with open(os.path.join(SAVE_DIR, volume_name.replace(".nii.gz", ".meta.txt")), "w") as f:
        f.write(str(item))

print("Done. 150 training samples saved.")
