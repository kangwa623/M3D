import os
import shutil
import re
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

def construct_nested_path(volume_name, split="train"):
    """
    Construct the nested path from filename.
    Example: train_1_a_1.nii.gz -> dataset/train/train_1/train_1_a/train_1_a_1.nii.gz
    """
    # Remove .nii.gz extension
    base_name = volume_name.replace(".nii.gz", "")
    
    # Pattern: train_<number>_<letter>_<number>
    # Extract: train_1_a_1 -> patient=1, phase=a, scan=1
    match = re.match(r'train_(\d+)_([a-z])_(\d+)', base_name)
    if match:
        patient_num = match.group(1)
        phase = match.group(2)
        scan_num = match.group(3)
        
        # Construct nested path
        nested_path = f"dataset/{split}/train_{patient_num}/train_{patient_num}_{phase}/{volume_name}"
        return nested_path
    
    # Try test pattern: test_<number>_<letter>_<number>
    match = re.match(r'test_(\d+)_([a-z])_(\d+)', base_name)
    if match:
        patient_num = match.group(1)
        phase = match.group(2)
        scan_num = match.group(3)
        
        nested_path = f"dataset/{split}/test_{patient_num}/test_{patient_num}_{phase}/{volume_name}"
        return nested_path
    
    # Fallback: try direct paths
    return None

def find_file_path_fast(repo_id, volume_name, split="train", repo_type="dataset"):
    """Try the correct nested path first, then fallback options."""
    # First, try the constructed nested path
    nested_path = construct_nested_path(volume_name, split)
    if nested_path:
        try:
            cached_file = hf_hub_download(
                repo_id=repo_id,
                filename=nested_path,
                repo_type=repo_type
            )
            return cached_file, nested_path
        except:
            pass
    
    # Fallback: try a few common patterns
    fallback_paths = [
        f"dataset/{split}_fixed/{volume_name}",
        f"dataset/{split}/{volume_name}",
        f"{split}_fixed/{volume_name}",
        f"{split}/{volume_name}"
    ]
    
    for path in fallback_paths:
        try:
            cached_file = hf_hub_download(
                repo_id=repo_id,
                filename=path,
                repo_type=repo_type
            )
            return cached_file, path
        except:
            continue
    
    return None, None

# ============================================================================
# PART 1: Download 150 samples from training set
# ============================================================================
print("=" * 80)
print("PART 1: Downloading 150 samples from TRAINING set (FAST VERSION)")
print("=" * 80)

print("Loading CT-RATE training metadata...")
ds_train = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading {N_TRAIN_SAMPLES} training .nii.gz files...")

train_count = 0
train_failed = []
train_success_path = None
total_processed = 0

for item in tqdm(ds_train, desc="Training samples"):
    total_processed += 1
    
    if train_count >= N_TRAIN_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]
        
        # 1. Save the Report
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TRAIN_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Try to download with correct nested path
        cached_file, used_path = find_file_path_fast(
            "ibrahimhamamci/CT-RATE",
            volume_name,
            split="train"
        )
        
        if cached_file is None:
            train_failed.append(volume_name)
            continue
        
        # Track which path worked
        if train_success_path is None:
            train_success_path = used_path.rsplit('/', 1)[0]
            print(f"\n✓ Found working path pattern: {train_success_path}")
        
        # 3. Copy the file
        img_path = os.path.join(SAVE_TRAIN_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        train_count += 1
        
        # Print progress every 10 files
        if train_count % 10 == 0:
            print(f"\n✓ Downloaded {train_count}/{N_TRAIN_SAMPLES} files (processed {total_processed} samples)")

    except Exception as e:
        train_failed.append(volume_name)
        continue

print(f"\n✓ Successfully downloaded {train_count} training samples!")
print(f"  Processed {total_processed} total samples from metadata")
if total_processed > 0:
    print(f"  Success rate: {train_count/total_processed*100:.1f}%")
print(f"  Images saved to: {SAVE_TRAIN_IMG_DIR}")
print(f"  Reports saved to: {SAVE_TRAIN_TXT_DIR}")

# ============================================================================
# PART 2: Download full test set
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: Downloading FULL TEST set")
print("=" * 80)

print("Loading CT-RATE test metadata...")
test_split_name = None
try:
    ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="test", streaming=True)
    test_split_name = "test"
except Exception as e1:
    try:
        ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="validation", streaming=True)
        test_split_name = "validation"
    except Exception as e2:
        try:
            ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="valid", streaming=True)
            test_split_name = "valid"
        except Exception as e3:
            print(f"Error: Could not find test/validation split.")
            raise

print(f"Using split: '{test_split_name}'")

# Count test samples
print("Counting test samples...")
test_items = []
for item in ds_test:
    test_items.append(item)
test_total = len(test_items)

print(f"Found {test_total} test samples. Downloading all available test .nii.gz files...")

# Reload for downloading
ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split=test_split_name, streaming=True)

test_count = 0
test_failed = []
test_success_path = None

for item in tqdm(ds_test, total=test_total, desc="Test samples"):
    try:
        volume_name = item["VolumeName"]
        
        # 1. Save the Report
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TEST_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Try to download with correct nested path
        cached_file, used_path = find_file_path_fast(
            "ibrahimhamamci/CT-RATE",
            volume_name,
            split=test_split_name
        )
        
        if cached_file is None:
            test_failed.append(volume_name)
            continue
        
        # Track which path worked
        if test_success_path is None:
            test_success_path = used_path.rsplit('/', 1)[0]
            print(f"\n✓ Found working path pattern: {test_success_path}")
        
        # 3. Copy the file
        img_path = os.path.join(SAVE_TEST_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        test_count += 1
        
        # Print progress every 10 files
        if test_count % 10 == 0:
            print(f"\n✓ Downloaded {test_count}/{test_total} test files")

    except Exception as e:
        test_failed.append(volume_name)
        continue

print(f"\n✓ Successfully downloaded {test_count}/{test_total} test samples!")
if test_failed:
    print(f"  Skipped {len(test_failed)} test files (not in repository)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"Training samples: {train_count} downloaded (requested: {N_TRAIN_SAMPLES})")
print(f"  Processed {total_processed} samples from metadata")
if total_processed > 0:
    print(f"  Success rate: {train_count/total_processed*100:.1f}%")
print(f"\nTest samples: {test_count}/{test_total} downloaded")
if train_success_path:
    print(f"\nWorking paths found:")
    print(f"  Training: {train_success_path}")
    if test_success_path:
        print(f"  Test: {test_success_path}")