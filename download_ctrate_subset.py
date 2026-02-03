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

# Alternative paths to try (in order of preference)
TRAIN_PATHS = [
    "dataset/train_fixed",
    "dataset/train",
    "train_fixed",
    "train"
]

TEST_PATHS = [
    "dataset/test_fixed",
    "dataset/valid_fixed",
    "dataset/test",
    "dataset/validation",
    "test_fixed",
    "valid_fixed",
    "test",
    "validation"
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
        except Exception as e:
            continue  # Try next path
    return None, None

# ============================================================================
# PART 1: Download 150 samples from training set
# ============================================================================
print("=" * 80)
print("PART 1: Downloading 150 samples from TRAINING set")
print("=" * 80)

print("Loading CT-RATE training metadata...")
# Try loading with the volumes config to get direct file access
try:
    # First try loading with volumes config which might have direct file access
    ds_train_volumes = load_dataset("ibrahimhamamci/CT-RATE", "volumes", split="train", streaming=True)
    use_volumes_config = True
    print("Using 'volumes' config for direct file access")
except:
    use_volumes_config = False
    print("Using 'reports' config (volumes config not available)")

ds_train = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading up to {N_TRAIN_SAMPLES} training .nii.gz files...")
print("Note: Some files may not exist in the repository and will be skipped.")

train_count = 0
train_failed = []
train_success_path = None  # Track which path worked
skipped_count = 0

for item in tqdm(ds_train, desc="Training samples"):
    if train_count >= N_TRAIN_SAMPLES:
        break
    
    try:
        volume_name = item["VolumeName"]  # This is the 'train_X.nii.gz' name
        
        # 1. Save the Report (as a .txt file) - always do this even if image fails
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TRAIN_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Try to download the .nii.gz file with multiple path options
        cached_file, used_path = find_file_path(
            "ibrahimhamamci/CT-RATE",
            volume_name,
            TRAIN_PATHS
        )
        
        if cached_file is None:
            # File doesn't exist - skip silently but count it
            skipped_count += 1
            train_failed.append(volume_name)
            # Only print every 10th failure to reduce noise
            if skipped_count % 10 == 0:
                print(f"\n⚠ Skipped {skipped_count} missing files so far (latest: {volume_name})")
            continue
        
        # Track which path worked (for future downloads)
        if train_success_path is None:
            train_success_path = used_path.rsplit('/', 1)[0]  # Get directory path
            print(f"\n✓ Found working path: {train_success_path}")
        
        # 3. Copy the .nii.gz file to your folder
        img_path = os.path.join(SAVE_TRAIN_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        train_count += 1

    except Exception as e:
        print(f"\nError downloading training sample {volume_name}: {e}")
        train_failed.append(volume_name)
        continue

# Continue trying until we get 150 successful downloads
if train_count < N_TRAIN_SAMPLES:
    print(f"\n⚠ Only found {train_count} available files out of {N_TRAIN_SAMPLES} requested.")
    print(f"  This is normal - not all files in metadata exist in the repository.")

print(f"\n✓ Successfully downloaded {train_count} training samples!")
if train_failed:
    print(f"⚠ Skipped {len(train_failed)} files that don't exist in repository")
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
            print(f"Error: Could not find test/validation split. Tried 'test', 'validation', and 'valid'.")
            print(f"Errors: {e1}, {e2}, {e3}")
            raise

print(f"Using split: '{test_split_name}'")

# First, we need to get the total count for progress bar
print("Counting test samples (this may take a moment)...")
test_items = []
for item in ds_test:
    test_items.append(item)
test_total = len(test_items)

print(f"Found {test_total} test samples. Downloading all available test .nii.gz files...")
print("Note: Some files may not exist in the repository and will be skipped.")

# Reload the dataset for actual downloading
ds_test = load_dataset("ibrahimhamamci/CT-RATE", "reports", split=test_split_name, streaming=True)

test_count = 0
test_failed = []
test_success_path = None
test_skipped_count = 0

for item in tqdm(ds_test, total=test_total, desc="Test samples"):
    try:
        volume_name = item["VolumeName"]  # This could be 'test_X.nii.gz' or similar
        
        # 1. Save the Report (as a .txt file) - always do this even if image fails
        report_content = f"FINDINGS:\n{item['Findings_EN']}\n\nIMPRESSIONS:\n{item['Impressions_EN']}"
        txt_filename = volume_name.replace(".nii.gz", ".txt")
        txt_path = os.path.join(SAVE_TEST_TXT_DIR, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # 2. Try to download the .nii.gz file with multiple path options
        cached_file, used_path = find_file_path(
            "ibrahimhamamci/CT-RATE",
            volume_name,
            TEST_PATHS
        )
        
        if cached_file is None:
            # File doesn't exist - skip silently but count it
            test_skipped_count += 1
            test_failed.append(volume_name)
            # Only print every 10th failure to reduce noise
            if test_skipped_count % 10 == 0:
                print(f"\n⚠ Skipped {test_skipped_count} missing test files so far (latest: {volume_name})")
            continue
        
        # Track which path worked
        if test_success_path is None:
            test_success_path = used_path.rsplit('/', 1)[0]
            print(f"\n✓ Found working path: {test_success_path}")
        
        # 3. Copy the .nii.gz file to your folder
        img_path = os.path.join(SAVE_TEST_IMG_DIR, volume_name)
        shutil.copy(cached_file, img_path)
        
        test_count += 1

    except Exception as e:
        print(f"\nError downloading test sample {volume_name}: {e}")
        test_failed.append(volume_name)
        continue

print(f"\n✓ Successfully downloaded {test_count}/{test_total} test samples!")
if test_failed:
    print(f"⚠ Skipped {len(test_failed)} test files that don't exist in repository")
print(f"  Images saved to: {SAVE_TEST_IMG_DIR}")
print(f"  Reports saved to: {SAVE_TEST_TXT_DIR}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"Training samples: {train_count} downloaded (requested: {N_TRAIN_SAMPLES})")
if train_failed:
    print(f"  Skipped {len(train_failed)} training files (not in repository)")
    if len(train_failed) <= 20:
        print(f"  Example skipped files: {', '.join(train_failed[:5])}...")
print(f"Test samples: {test_count}/{test_total} downloaded")
if test_failed:
    print(f"  Skipped {len(test_failed)} test files (not in repository)")
    if len(test_failed) <= 20:
        print(f"  Example skipped files: {', '.join(test_failed[:5])}...")
print(f"\nWorking paths found:")
if train_success_path:
    print(f"  Training: {train_success_path}")
if test_success_path:
    print(f"  Test: {test_success_path}")
print(f"\nDirectory structure:")
print(f"  Training images: {SAVE_TRAIN_IMG_DIR}")
print(f"  Training reports: {SAVE_TRAIN_TXT_DIR}")
print(f"  Test images: {SAVE_TEST_IMG_DIR}")
print(f"  Test reports: {SAVE_TEST_TXT_DIR}")
print(f"\nNote: It's normal for some files to be missing from the repository.")
print(f"      Reports are saved for all samples, even if images are missing.")
