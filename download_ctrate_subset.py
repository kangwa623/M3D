import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- CONFIGURATION ---
N_TRAIN_SAMPLES = 150  # Number of training samples to download
SAVE_TRAIN_IMG_DIR = "ctrate_volumes/train"
SAVE_TRAIN_TXT_DIR = "ctrate_reports/train"
SAVE_TEST_IMG_DIR = "ctrate_volumes/test"
SAVE_TEST_TXT_DIR = "ctrate_reports/test"

# Create local folders
os.makedirs(SAVE_TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TRAIN_TXT_DIR, exist_ok=True)
os.makedirs(SAVE_TEST_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_TEST_TXT_DIR, exist_ok=True)

# ============================================================================
# PART 1: Download 150 samples from training set
# ============================================================================
print("=" * 80)
print("PART 1: Downloading 150 samples from TRAINING set")
print("=" * 80)

print("Loading CT-RATE training metadata...")
# Load the 'reports' config to get the mapping of VolumeNames for training set
ds_train = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading {N_TRAIN_SAMPLES} training .nii.gz files...")

train_count = 0
for item in tqdm(ds_train, desc="Training samples"):
    if train_count >= N_TRAIN_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]  # This is the 'train_X.nii.gz' name
        
        # 1. Save the Report (as a .txt file)
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TRAIN_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Download the actual .nii.gz Volume from train_fixed
        cached_file = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=f"dataset/train_fixed/{volume_name}", 
            repo_type="dataset"
        )
        
        # 3. Copy the .nii.gz file to your folder
        img_path = os.path.join(SAVE_TRAIN_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        train_count += 1

    except Exception as e:
        print(f"\nError downloading training sample {volume_name}: {e}")
        continue

print(f"\n✓ Successfully downloaded {train_count} training samples!")
print(f"  Images saved to: {SAVE_TRAIN_IMG_DIR}")
print(f"  Reports saved to: {SAVE_TRAIN_TXT_DIR}")

# ============================================================================
# PART 2: Download full test set
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: Downloading FULL TEST set")
print("=" * 80)

print("Loading CT-RATE test metadata...")
# Load the test split - try 'test' first, if not available try 'validation' or 'valid'
try:
    ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="test", streaming=True)
    test_split_name = "test"
    dataset_path = "dataset/test_fixed"
except Exception as e1:
    try:
        ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="validation", streaming=True)
        test_split_name = "validation"
        dataset_path = "dataset/valid_fixed"
    except Exception as e2:
        try:
            ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="valid", streaming=True)
            test_split_name = "valid"
            dataset_path = "dataset/valid_fixed"
        except Exception as e3:
            print(f"Error: Could not find test/validation split. Tried 'test', 'validation', and 'valid'.")
            print(f"Errors: {e1}, {e2}, {e3}")
            raise

print(f"Using split: '{test_split_name}' with dataset path: '{dataset_path}'")

# First, we need to get the total count for progress bar
# Since streaming=True, we'll count as we go
print("Counting test samples (this may take a moment)...")
test_items = []
for item in ds_test:
    test_items.append(item)
test_total = len(test_items)

print(f"Found {test_total} test samples. Downloading all test .nii.gz files...")

# Reload the dataset for actual downloading
ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split=test_split_name, streaming=True)

test_count = 0
for item in tqdm(ds_test, total=test_total, desc="Test samples"):
    try:
        volume_name = item["VolumeName"]  # This could be 'test_X.nii.gz' or similar
        
        # 1. Save the Report (as a .txt file)
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TEST_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Download the actual .nii.gz Volume from test_fixed or valid_fixed
        cached_file = hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename=f"{dataset_path}/{volume_name}", 
            repo_type="dataset"
        )
        
        # 3. Copy the .nii.gz file to your folder
        img_path = os.path.join(SAVE_TEST_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        test_count += 1

    except Exception as e:
        print(f"\nError downloading test sample {volume_name}: {e}")
        continue

print(f"\n✓ Successfully downloaded {test_count} test samples!")
print(f"  Images saved to: {SAVE_TEST_IMG_DIR}")
print(f"  Reports saved to: {SAVE_TEST_TXT_DIR}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"Training samples: {train_count}/{N_TRAIN_SAMPLES}")
print(f"Test samples: {test_count}/{test_total}")
print(f"\nAll files downloaded successfully!")
print(f"\nDirectory structure:")
print(f"  Training images: {SAVE_TRAIN_IMG_DIR}")
print(f"  Training reports: {SAVE_TRAIN_TXT_DIR}")
print(f"  Test images: {SAVE_TEST_IMG_DIR}")
print(f"  Test reports: {SAVE_TEST_TXT_DIR}")
