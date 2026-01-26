import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from multiprocessing import Pool
from tqdm import tqdm

# =========================
# CONFIGURATION (ct-rate-mini)
# =========================
BASE_DIR = "/nfs/usrhome2/africanstu/kangwa/m3d/M3D/datasets/ct-rate-mini"

# Where the raw .nii.gz files live
NII_DIR = os.path.join(BASE_DIR, "data", "images")

# Where M3D-ready samples will be written
OUTPUT_DIR = os.path.join(BASE_DIR, "m3d_ready")

TARGET_SIZE = (256, 256)
TARGET_FRAMES = 32
TARGET_SPACING = (1.5, 0.75, 0.75)  # (Z, X, Y) in mm
HU_MIN, HU_MAX = -150, 200
NUM_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


def reorient_to_ras(img):
    orig_ornt = nib.io_orientation(img.affine)
    targ_ornt = nib.orientations.axcodes2ornt('RAS')
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    return img.as_reoriented(transform)


def resize_array(array, current_spacing, target_spacing):
    """Physically resample the volume using trilinear interpolation."""
    # array: [1, 1, D, H, W]
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(3)]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(3)]
    resized = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    return resized.squeeze().cpu().numpy()


def process_single_file(filename):
    try:
        # For mini dataset, treat everything as venous
        phase_part = "venous"
        p_id_str = os.path.splitext(os.path.splitext(filename)[0])[0]

        patient_folder = os.path.join(OUTPUT_DIR, p_id_str)
        os.makedirs(patient_folder, exist_ok=True)

        # Create required text file
        txt_path = os.path.join(patient_folder, f"{p_id_str}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w") as f:
                f.write("CT scan sample from ct-rate-mini dataset.")

        path = os.path.join(NII_DIR, filename)
        img = nib.load(path)
        img = reorient_to_ras(img)
        data = img.get_fdata()
        zooms = img.header.get_zooms()
        current_spacing = (zooms[2], zooms[1], zooms[0])

        # Reorder to (D, H, W)
        data = data.transpose(2, 1, 0)

        # Window + normalize
        data = np.clip(data, HU_MIN, HU_MAX)
        data = (data - HU_MIN) / (HU_MAX - HU_MIN)
        data = data.astype(np.float32)

        # Physical resampling
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        resampled_data = resize_array(tensor, current_spacing, TARGET_SPACING)

        # Spatial resize to 256x256
        final_data = zoom(
            resampled_data,
            (1,
             TARGET_SIZE[0] / resampled_data.shape[1],
             TARGET_SIZE[1] / resampled_data.shape[2]),
            order=1
        )

        # Temporal resize to 32 slices
        final_data = zoom(final_data, (TARGET_FRAMES / final_data.shape[0], 1, 1), order=1)

        # Save as (1, 32, 256, 256)
        output_path = os.path.join(patient_folder, "venous.npy")
        np.save(output_path, final_data[np.newaxis, ...])

        return f"Success: {filename}"
    except Exception as e:
        return f"Error: {filename} | {str(e)}"


if __name__ == "__main__":
    all_files = [f for f in os.listdir(NII_DIR) if f.endswith(".nii.gz")]
    print(f"Processing {len(all_files)} files using {NUM_WORKERS} processes...")

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, all_files),
                            total=len(all_files)))

    errors = [r for r in results if "Error" in r]
    print(f"\nDone! Processed {len(results) - len(errors)} successfully. Errors: {len(errors)}")