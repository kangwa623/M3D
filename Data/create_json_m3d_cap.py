import pandas as pd
import json
import os

# --- CONFIGURATION ---
# The root folder where your processed patient folders (0, 1, 2...) are located
base_data_path = "/nfs/usrhome2/mkfmelbatel/datasets/trials_report/m3d_npy_v1"

# List of split files to process
splits = ["train.csv", "val.csv", "test.csv"]

# The final dictionary structure
final_data = {}


def create_split_list(csv_path):
    """Reads a CSV and creates a list of dicts for the JSON."""
    split_list = []
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        p_id = str(row['patient_id'])

        # Define paths based on your folder structure
        # image: patient_id/venous.npy
        # text:  patient_id/patient_id.txt
        image_rel_path = f"{p_id}/venous.npy"
        text_rel_path = f"{p_id}/{p_id}.txt"

        # Verify files exist before adding (optional but recommended)
        full_img_path = os.path.join(base_data_path, image_rel_path)
        full_txt_path = os.path.join(base_data_path, text_rel_path)

        if os.path.exists(full_img_path) and os.path.exists(full_txt_path):
            split_list.append({
                "image": image_rel_path,
                "text": text_rel_path
            })
        else:
            print(f"Warning: Missing files for patient {p_id}. Skipping...")

    return split_list


# --- EXECUTION ---
for split_file in splits:
    split_name = split_file.split('.')[0]  # e.g., 'train'
    print(f"Processing {split_name}...")
    final_data[split_name] = create_split_list(split_file)

# Save to a single JSON file
output_json = "m3d_dataset_split.json"
with open(output_json, 'w') as f:
    json.dump(final_data, f, indent=2)

print(f"\nSuccessfully created {output_json}")
print(f"Total training samples: {len(final_data['train'])}")