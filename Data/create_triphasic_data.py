import pandas as pd
import json
import os

# --- CONFIGURATION ---
# The root folder where your processed patient folders (0, 1, 2...) are located
base_data_path = "/nfs/usrhome2/mkfmelbatel/datasets/trials_report/m3d_npy_v1"

# List of split files to process
splits = ["train.csv", "val.csv", "test.csv"]

# Define the phases you want to include
PHASES = ["arterial.npy", "venous.npy", "delayed.npy"]

# The final dictionary structure
final_data = {}


def create_split_list(csv_path):
    """Reads a CSV and creates a list of dicts with 3-phase image paths."""
    split_list = []
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        p_id = str(row['patient_id'])

        # 1. Construct relative paths for the 3 phases
        image_paths = [f"{p_id}/{phase}" for phase in PHASES]
        text_rel_path = f"{p_id}/{p_id}.txt"

        # 2. Verify all 3 phase files and the text file exist
        all_phases_exist = all(os.path.exists(os.path.join(base_data_path, p)) for p in image_paths)
        full_txt_path = os.path.join(base_data_path, text_rel_path)

        if all_phases_exist and os.path.exists(full_txt_path):
            split_list.append({
                "image": image_paths,  # Storing as a list for the multi-phase __getitem__
                "text": text_rel_path
            })
        else:
            # Troubleshooting: Find out which phase is missing
            missing = [p for p in image_paths if not os.path.exists(os.path.join(base_data_path, p))]
            if not os.path.exists(full_txt_path): missing.append("text_file")
            print(f"Warning: Missing files for patient {p_id}: {missing}. Skipping...")

    return split_list


# --- EXECUTION ---
for split_file in splits:
    split_name = split_file.split('.')[0]  # e.g., 'train'
    if not os.path.exists(split_file):
        print(f"File {split_file} not found. Skipping...")
        continue

    print(f"Processing {split_name}...")
    final_data[split_name] = create_split_list(split_file)

# Save to a single JSON file
output_json = "m3d_triphasic_dataset.json"
with open(output_json, 'w') as f:
    json.dump(final_data, f, indent=2)

print(f"\nSuccessfully created {output_json}")
print(f"Total training samples: {len(final_data.get('train', []))}")