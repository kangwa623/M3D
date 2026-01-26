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

# --- CONFIGURATION (your real folders) ---
NII_DIR = "/home/africanstu/kangwa/m3d/M3D/datasets/ct-rate-mini/data/images"
CSV_PATH = "/home/africanstu/kangwa/m3d/M3D/datasets/ct-rate-mini/meta.csv"
JSON_DIR = "/home/africanstu/kangwa/m3d/M3D/datasets/ct-rate-mini/reports"
OUTPUT_DIR = "/home/africanstu/kangwa/m3d/M3D/datasets/ct-rate-mini/m3d_npy"

# Processing Settings
TARGET_SIZE = (256, 256)
TARGET_FRAMES = 32
TARGET_SPACING = (1.5, 0.75, 0.75)
HU_MIN, HU_MAX = -150, 200
NUM_WORKERS = 8

print("NII_DIR:", NII_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV is optional for mini dataset
df_meta = None
if os.path.exists(CSV_PATH):
    df_meta = pd.read_csv(CSV_PATH)

# JSON index is optional
json_map = {}
if os.path.isdir(JSON_DIR):
    print("Indexing JSON files...")
    for f in os.listdir(JSON_DIR):
        if f.endswith(".json"):
            p_id_prefix = f.split("_")[0].replace(".json", "")
            json_map[p_id_prefix] = f


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


def process_single_file(filename):
    try:
        # For ct-rate-mini we treat everything as venous
        phase_part = "venous"
        p_id_str = os.path.splitext(os.path.splitext(filename)[0])[0]

        patient_folder = os.path.join(OUTPUT_DIR, p_id_str)
        os.makedirs(patient_folder, exist_ok=True)

        # Text file (required by M3D)
        txt_path = os.path.join(patient_folder, f"{p_id_str}.txt")
        if not os.path.exists(txt_path):
            if p_id_str in json_map:
                json_path = os.path.join(JSON_DIR, json_map[p_id_str])
                with open(json_path, "r") as f:
                    data = json.load(f)
                    findings = data.get("findings", data.get("report", ""))
            else:
                findings = "CT scan sample from ct-rate-mini."
            with open(txt_path, "w") as f:
                f.write(findings.strip())

        path = os.path.join(NII_DIR, filename)
        img = nib.load(path)
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
        return f"Error: {filename} | {str(e)}"


if __name__ == "__main__":
    if not os.path.isdir(NII_DIR):
        raise RuntimeError(f"NII_DIR does not exist: {NII_DIR}")

    all_files = [f for f in os.listdir(NII_DIR) if f.endswith(".nii.gz")]
    print(f"Processing {len(all_files)} files using {NUM_WORKERS} processes...")

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, all_files), total=len(all_files)))

    errors = [r for r in results if "Error" in r]
    print(f"\nDone! Processed {len(results) - len(errors)} successfully. Errors: {len(errors)}")