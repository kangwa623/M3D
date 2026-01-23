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

# --- CONFIGURATION ---
NII_DIR = "/nfs/usrhome2/mkfmelbatel/datasets/trials_report/raw/processed_dataset"
CSV_PATH = "/nfs/usrhome2/mkfmelbatel/datasets/trials_report/raw/final_patient_series_selection.csv"
JSON_DIR = "/nfs/usrhome2/mkfmelbatel/datasets/trials_report/raw/cleaned_reports_final_v1"
# --- EDITED CONFIGURATION ---
OUTPUT_DIR = "/nfs/usrhome2/mkfmelbatel/datasets/trials_report/m3d_npy_v1"

# Recommended: Liver/Abdominal Window
def reorient_to_ras(img):
    orig_ornt = nib.io_orientation(img.affine)
    targ_ornt = nib.orientations.axcodes2ornt('RAS')
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    return img.as_reoriented(transform)
# Map -150 to -1.0 and 250 to 1.0
# Processing Settings
TARGET_SIZE = (256, 256) # From 512 to 256
TARGET_FRAMES = 32  # Number of slices in output video
TARGET_SPACING = (1.5, 0.75, 0.75)  # (Z, X, Y) in mm
HU_MIN, HU_MAX = -150, 200  # Window range
NUM_WORKERS = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_meta = pd.read_csv(CSV_PATH)


def center_crop(image, target_size=(512, 512)):
    """Square center crop based on the smallest spatial dimension."""
    d, h, w = image.shape
    crop_size = min(h, w)

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    return image[:, top:top + crop_size, left:left + crop_size]


def resize_array(array, current_spacing, target_spacing):
    """Physically resample the volume using trilinear interpolation."""
    # array: [1, 1, D, H, W]
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(3)]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(3)]

    resized = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    return resized.squeeze().cpu().numpy()

# --- GLOBAL JSON INDEX ---
# We map the ID prefix to the actual filename (e.g., "10" -> "10_9116_2.json")
print("Indexing JSON files...")
json_map = {}
for f in os.listdir(JSON_DIR):
    if f.endswith('.json'):
        # Split by underscore to get the ID (the part before the random numbers)
        p_id_prefix = f.split('_')[0].replace('.json', '')
        json_map[p_id_prefix] = f

def process_single_file(filename):
    try:
        # 1. Parse Phase and Patient ID (e.g., 'arterial_10.nii.gz')
        phase_part = filename.split('_')[0]
        p_id_str = filename.split('_')[-1].replace('.nii.gz', '')
        p_id = int(p_id_str)
        # 2. Create Patient Folder
        patient_folder = os.path.join(OUTPUT_DIR, p_id_str)
        os.makedirs(patient_folder, exist_ok=True)

        # 3. UPDATED TEXT PROCESSING (Handle ID_randomnumber.json)
        txt_path = os.path.join(patient_folder, f"{p_id_str}.txt")
        if not os.path.exists(txt_path):
            actual_json_name = json_map.get(p_id_str)
            if actual_json_name:
                json_path = os.path.join(JSON_DIR, actual_json_name)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    findings = data.get("findings", data.get("report", ""))
                with open(txt_path, 'w') as f:
                    f.write(findings.strip())        # 2. Get Metadata for this patient/phase
        # Map filenames to CSV column prefixes
        phase_map = {
            'arterial': 'Arterial_Thin',
            'venous': 'Venous_Thin',
            'delayed': 'Delayed_Thin',
            'non_contrast': 'Non Contrast_Thin'
        }
        prefix = phase_map.get(phase_part)
        if not prefix: return f"Skip: Unknown phase {phase_part}"

        row = df_meta[df_meta['patient_id'] == p_id]
        if row.empty: return f"Skip: No CSV entry for ID {p_id}"

        slope = float(row[f"{prefix}_RescaleSlope"].iloc[0])
        intercept = float(row[f"{prefix}_RescaleIntercept"].iloc[0])

        # 1-3. [Keep your existing ID parsing, Metadata, and NIfTI Loading logic]
        path = os.path.join(NII_DIR, filename)
        img = nib.load(path)
        img = reorient_to_ras(img)
        data = img.get_fdata()
        zooms = img.header.get_zooms()
        current_spacing = (zooms[2], zooms[1], zooms[0])

        # 4. Apply HU Calibration & Normalization (0.0 to 1.0)
        # data = (slope * data + intercept).transpose(2, 1, 0)
        data = data.transpose(2, 1, 0)

        data = np.clip(data, HU_MIN, HU_MAX)
        data = (data - HU_MIN) / (HU_MAX - HU_MIN)
        data = data.astype(np.float32)

        # 5. Physical Resampling
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        resampled_data = resize_array(tensor, current_spacing, TARGET_SPACING)

        # 6. Spatial Resize to 256x256 & Orientation Flip
        final_data = zoom(resampled_data, (1, TARGET_SIZE[0] / resampled_data.shape[1],
                                           TARGET_SIZE[1] / resampled_data.shape[2]), order=1)
        final_data = np.flip(final_data, axis=(1, 2))

        final_data = zoom(final_data, (TARGET_FRAMES / final_data.shape[0], 1, 1), order=1)
        # # 7. Temporal Sampling (Select exactly 32 frames)
        # if final_data.shape[0] > TARGET_FRAMES:
        #     indices = np.linspace(0, final_data.shape[0] - 1, TARGET_FRAMES, dtype=int)
        #     final_data = final_data[indices]
        # elif final_data.shape[0] < TARGET_FRAMES:
        #     # If volume is too thin, pad with zeros or interpolate to 32
        #     final_data = zoom(final_data, (TARGET_FRAMES / final_data.shape[0], 1, 1), order=1)

        # 8. Final Shape Adjustment (1, 32, 256, 256) and Save
        output_path = os.path.join(patient_folder, f"{phase_part}.npy")
        np.save(output_path, final_data[np.newaxis, ...])

        return f"Success: {filename}"
    except Exception as e:
        return f"Error: {filename} | {str(e)}"
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    all_files = [f for f in os.listdir(NII_DIR) if f.endswith('.nii.gz')]
    print(f"Processing {len(all_files)} files using {NUM_WORKERS} processes...")

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, all_files), total=len(all_files)))

    errors = [r for r in results if "Error" in r]
    print(f"\nDone! Processed {len(results) - len(errors)} successfully. Errors: {len(errors)}")