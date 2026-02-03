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

# Comprehensive path options to try (based on common CT-RATE structures)
TRAIN_PATHS = [
    "dataset/train_fixed",
    "dataset/train",
    "train_fixed",
    "train",
    "volumes/train_fixed",
    "volumes/train",
    "data/train_fixed",
    "data/train"
]

TEST_PATHS = [
    "dataset/test_fixed",
    "dataset/valid_fixed",
    "dataset/test",
    "dataset/validation",
    "test_fixed",
    "valid_fixed",
    "test",
    "validation",
    "volumes/test_fixed",
    "volumes/valid_fixed",
    "data/test_fixed",
    "data/valid_fixed"
]

def find_file_path(repo_id, volume_name, path_options, repo_type="dataset"):
    """Try multiple path options to find the file."""
    for path_option in path_options:
        try:
            full_path = f"{path_option}/{volume_name}"
            cached_file = hf_hub_download(
                repo_id=repo_id,
                filename=full_path,
                repo_type=repo_type
            )
            return cached_file, full_path
        except Exception:
            continue  # Try next path
    return None, None

# ============================================================================
# PART 1: Download 150 samples from training set
# ============================================================================
print("=" * 80)
print("PART 1: Downloading 150 samples from TRAINING set")
print("=" * 80)

print("Loading CT-RATE training metadata...")
ds_train = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading up to {N_TRAIN_SAMPLES} training .nii.gz files...")
print("Note: Will continue processing until {N_TRAIN_SAMPLES} files are found or dataset is exhausted.")

train_count = 0
train_failed = []
train_success_path = None
skipped_count = 0
total_processed = 0

for item in tqdm(ds_train, desc="Training samples"):
    total_processed += 1
    
    if train_count >= N_TRAIN_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]
        
        # 1. Save the Report (as a .txt file) - always do this
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TRAIN_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Try to download the .nii.gz file
        cached_file, used_path = find_file_path(
            "ibrahimhamamci/CT-RATE",
            volume_name,
            TRAIN_PATHS
        )
        
        if cached_file is None:
            skipped_count += 1
            train_failed.append(volume_name)
            # Only print every 50th failure to reduce noise
            if skipped_count % 50 == 0:
                print(f"\n⚠ Processed {total_processed} samples, found {train_count} files, skipped {skipped_count} missing files")
            continue
        
        # Track which path worked and prioritize it
        if train_success_path is None:
            train_success_path = used_path.rsplit('/', 1)[0]
            print(f"\n✓ Found working path: {train_success_path}")
            # Move working path to front of list for faster future downloads
            if train_success_path in TRAIN_PATHS:
                TRAIN_PATHS.remove(train_success_path)
            TRAIN_PATHS.insert(0, train_success_path)
        
        # 3. Copy the .nii.gz file to your folder
        img_path = os.path.join(SAVE_TRAIN_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        train_count += 1

    except Exception as e:
        print(f"\nError processing {volume_name}: {e}")
        train_failed.append(volume_name)
        continue

print(f"\n✓ Successfully downloaded {train_count} training samples!")
print(f"  Processed {total_processed} total samples from metadata")
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
test_skipped_count = 0

for item in tqdm(ds_test, total=test_total, desc="Test samples"):
    try:
        volume_name = item["VolumeName"]
        
        # 1. Save the Report
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TEST_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Try to download the .nii.gz file
        cached_file, used_path = find_file_path(
            "ibrahimhamamci/CT-RATE",
            volume_name,
            TEST_PATHS
        )
        
        if cached_file is None:
            test_skipped_count += 1
            test_failed.append(volume_name)
            if test_skipped_count % 50 == 0:
                print(f"\n⚠ Processed {test_count + test_skipped_count} test samples, found {test_count} files")
            continue
        
        # Track which path worked
        if test_success_path is None:
            test_success_path = used_path.rsplit('/', 1)[0]
            print(f"\n✓ Found working path: {test_success_path}")
            if test_success_path in TEST_PATHS:
                TEST_PATHS.remove(test_success_path)
            TEST_PATHS.insert(0, test_success_path)
        
        # 3. Copy the file
        img_path = os.path.join(SAVE_TEST_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        test_count += 1

    except Exception as e:
        print(f"\nError processing test sample {volume_name}: {e}")
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
print(f"\nDirectory structure:")
print(f"  Training images: {SAVE_TRAIN_IMG_DIR}")
print(f"  Training reports: {SAVE_TRAIN_TXT_DIR}")
print(f"  Test images: {SAVE_TEST_IMG_DIR}")
print(f"  Test reports: {SAVE_TEST_TXT_DIR}")