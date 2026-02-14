import json
import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
from scipy.ndimage import zoom
from multiprocessing import Pool
from tqdm import tqdm

# --- CONFIGURATION (Updated for your new paths) ---
BASE_DIR = "/nfs/usrhome2/africanstu/kangwa/m3d/M3D"
NII_TRAIN_DIR = os.path.join(BASE_DIR, "ctrate_volumes/train")
NII_TEST_DIR = os.path.join(BASE_DIR, "ctrate_volumes/test")
REPORT_TRAIN_DIR = os.path.join(BASE_DIR, "ctrate_reports/train")
REPORT_TEST_DIR = os.path.join(BASE_DIR, "ctrate_reports/test")
OUTPUT_TRAIN_DIR = os.path.join(BASE_DIR, "ctrate_volumes/m3d_npy/train")
OUTPUT_TEST_DIR = os.path.join(BASE_DIR, "ctrate_volumes/m3d_npy/test")

# Processing Settings
TARGET_SIZE = (256, 256)
TARGET_FRAMES = 32
TARGET_SPACING = (1.5, 0.75, 0.75)
HU_MIN, HU_MAX = -150, 200
NUM_WORKERS = 16  # Increased for faster processing

def reorient_to_ras(img):
    orig_ornt = nib.io_orientation(img.affine)
    targ_ornt = nib.orientations.axcodes2ornt("RAS")
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    return img.as_reoriented(transform)

def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(3)]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(3)]
    resized = F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
    return resized.squeeze().cpu().numpy()

def process_single_file(args):
    """Process a single file - updated to handle both train and test"""
    nii_path, report_dir, output_dir, split = args
    try:
        filename = os.path.basename(nii_path)
        # Extract patient ID from filename (e.g., train_1_a_1.nii.gz -> train_1_a_1)
        p_id_str = os.path.splitext(os.path.splitext(filename)[0])[0]
        phase_part = "venous"  # CT-RATE is typically venous phase
        
        patient_folder = os.path.join(output_dir, p_id_str)
        os.makedirs(patient_folder, exist_ok=True)
        
        # Copy or create text file from reports
        txt_path = os.path.join(patient_folder, f"{p_id_str}.txt")
        report_path = os.path.join(report_dir, f"{p_id_str}.txt")
        
        if os.path.exists(report_path):
            # Copy the report
            import shutil
            shutil.copy(report_path, txt_path)
        elif not os.path.exists(txt_path):
            # Create a default report
            with open(txt_path, "w") as f:
                f.write(f"CT scan sample from CT-RATE {split} set.")
        
        # Process the image
        img = nib.load(nii_path)
        img = reorient_to_ras(img)
        data = img.get_fdata()
        zooms = img.header.get_zooms()
        current_spacing = (zooms[2], zooms[1], zooms[0])
        
        data = data.transpose(2, 1, 0)
        data = np.clip(data, HU_MIN, HU_MAX)
        data = (data - HU_MIN) / (HU_MAX - HU_MIN)
        data = data.astype(np.float32)
        
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        resampled_data = resize_array(tensor, current_spacing, TARGET_SPACING)
        
        final_data = zoom(
            resampled_data,
            (1, TARGET_SIZE[0] / resampled_data.shape[1],
             TARGET_SIZE[1] / resampled_data.shape[2]),
            order=1
        )
        
        final_data = zoom(final_data, (TARGET_FRAMES / final_data.shape[0], 1, 1), order=1)
        
        output_path = os.path.join(patient_folder, f"{phase_part}.npy")
        np.save(output_path, final_data[np.newaxis, ...])
        
        return f"Success: {filename}"
    except Exception as e:
        return f"Error: {os.path.basename(nii_path)} | {str(e)}"

def process_split(nii_dir, report_dir, output_dir, split_name):
    """Process a single split (train or test)"""
    print(f"\n{'='*80}")
    print(f"Processing {split_name.upper()} set")
    print(f"{'='*80}")
    print(f"Input: {nii_dir}")
    print(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isdir(nii_dir):
        print(f"Warning: {nii_dir} does not exist. Skipping {split_name} set.")
        return
    
    all_files = [os.path.join(nii_dir, f) for f in os.listdir(nii_dir) if f.endswith(".nii.gz")]
    print(f"Found {len(all_files)} files to process")
    print(f"Using {NUM_WORKERS} worker processes...")
    
    # Prepare arguments for each file
    args_list = [(f, report_dir, output_dir, split_name) for f in all_files]
    
    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, args_list), total=len(all_files)))
    
    errors = [r for r in results if "Error" in r]
    successes = len(results) - len(errors)
    print(f"\n✓ {split_name.upper()} set complete!")
    print(f"  Successfully processed: {successes}/{len(all_files)}")
    if errors:
        print(f"  Errors: {len(errors)}")
        print(f"  First few errors:")
        for err in errors[:5]:
            print(f"    {err}")

if __name__ == "__main__":
    print("="*80)
    print("CT-RATE Preprocessing Script")
    print("="*80)
    
    # Process training set
    process_split(NII_TRAIN_DIR, REPORT_TRAIN_DIR, OUTPUT_TRAIN_DIR, "train")
    
    # Process test set
    process_split(NII_TEST_DIR, REPORT_TEST_DIR, OUTPUT_TEST_DIR, "test")
    
    print("\n" + "="*80)
    print("ALL PREPROCESSING COMPLETE!")
    print("="*80)