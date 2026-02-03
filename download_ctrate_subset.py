import os
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
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
# STEP 1: Explore repository structure to find actual file locations
# ============================================================================
print("=" * 80)
print("STEP 1: Exploring repository structure...")
print("=" * 80)

def explore_repo_structure():
    """List files in the repository to understand the structure."""
    print("Listing repository files (this may take a moment)...")
    try:
        files = list(list_repo_files(
            repo_id="ibrahimhamamci/CT-RATE",
            repo_type="dataset"
        ))
        
        # Filter for .nii.gz files and group by directory
        nii_files = [f for f in files if f.endswith('.nii.gz')]
        directories = {}
        for f in nii_files[:100]:  # Sample first 100 to understand structure
            parts = f.split('/')
            if len(parts) > 1:
                dir_name = '/'.join(parts[:-1])
                if dir_name not in directories:
                    directories[dir_name] = []
                directories[dir_name].append(parts[-1])
        
        print(f"\nFound {len(nii_files)} .nii.gz files in repository")
        print(f"Sample directories found:")
        for dir_name, file_list in list(directories.items())[:5]:
            print(f"  {dir_name}: {len(file_list)} files (e.g., {file_list[0]})")
        
        return directories, nii_files
    except Exception as e:
        print(f"Could not list repository files: {e}")
        print("Will proceed with standard path attempts...")
        return None, None

repo_dirs, all_nii_files = explore_repo_structure()

# Build path options based on what we found
if repo_dirs:
    # Use discovered directories
    TRAIN_PATHS = list(repo_dirs.keys())[:10]  # Try top 10 discovered paths
    TEST_PATHS = list(repo_dirs.keys())[:10]
    print(f"\nUsing discovered paths for download attempts")
else:
    # Fallback to standard paths
    TRAIN_PATHS = [
        "dataset/train_fixed",
        "dataset/train",
        "train_fixed",
        "train",
        "volumes/train_fixed",
        "volumes/train"
    ]
    TEST_PATHS = [
        "dataset/test_fixed",
        "dataset/valid_fixed",
        "dataset/test",
        "dataset/validation",
        "test_fixed",
        "valid_fixed",
        "volumes/test_fixed",
        "volumes/valid_fixed"
    ]

# Create a set of available files for quick lookup
available_files = set()
if all_nii_files:
    available_files = {f.split('/')[-1] for f in all_nii_files}  # Just filenames
    print(f"Created lookup set with {len(available_files)} available files")

def find_file_path(repo_id, volume_name, path_options, repo_type="dataset"):
    """Try multiple path options to find the file."""
    # Quick check: if we have a lookup set, check if file exists first
    if available_files and volume_name not in available_files:
        return None, None
    
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
print("\n" + "=" * 80)
print("PART 2: Downloading 150 samples from TRAINING set")
print("=" * 80)

print("Loading CT-RATE training metadata...")
ds_train = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

print(f"Downloading up to {N_TRAIN_SAMPLES} training .nii.gz files...")

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
        
        # Track which path worked
        if train_success_path is None:
            train_success_path = used_path.rsplit('/', 1)[0]
            print(f"\n✓ Found working path: {train_success_path}")
            # Update path list to prioritize this path
            if train_success_path not in TRAIN_PATHS:
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
print(f"  Skipped {len(train_failed)} files that don't exist in repository")
print(f"  Images saved to: {SAVE_TRAIN_IMG_DIR}")
print(f"  Reports saved to: {SAVE_TRAIN_TXT_DIR}")

# ============================================================================
# PART 2: Download full test set
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: Downloading FULL TEST set")
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
            if test_success_path not in TEST_PATHS:
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
print(f"  Success rate: {train_count/total_processed*100:.1f}%")
if train_failed and len(train_failed) <= 20:
    print(f"  Example skipped: {', '.join(train_failed[:3])}...")
print(f"\nTest samples: {test_count}/{test_total} downloaded")
if test_success_path:
    print(f"\nWorking paths:")
    print(f"  Training: {train_success_path}")
    print(f"  Test: {test_success_path}")
