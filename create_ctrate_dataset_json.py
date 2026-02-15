# create_ctrate_dataset_json.py
import os
import json

# Use relative or absolute paths; adjust for your machine (Windows: use your M3D path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # or e.g. r"C:\Users\kangw\Documents\AFRICAI\M3D"
data_root = os.path.join(BASE_DIR, "ctrate_volumes", "m3d_npy")
train_dir = os.path.join(data_root, "train")
test_dir = os.path.join(data_root, "test")
output_json = os.path.join(BASE_DIR, "Data", "ctrate_dataset.json")

VAL_RATIO = 0.1  # 10% of train used as validation

os.makedirs(os.path.dirname(output_json), exist_ok=True)

def collect_split(split_dir, prefix):
    out = []
    if not os.path.isdir(split_dir):
        return out
    for patient_id in sorted(os.listdir(split_dir)):
        patient_dir = os.path.join(split_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        img_path = os.path.join(patient_dir, "venous.npy")
        txt_path = os.path.join(patient_dir, f"{patient_id}.txt")
        if os.path.isfile(img_path) and os.path.isfile(txt_path):
            out.append({
                "image": f"{prefix}/{patient_id}/venous.npy",
                "text": f"{prefix}/{patient_id}/{patient_id}.txt"
            })
    return out

train_list = collect_split(train_dir, "train")
test_list = collect_split(test_dir, "test")

# Split train into train/val (last VAL_RATIO as val)
n_val = max(1, int(len(train_list) * VAL_RATIO))
val_list = train_list[-n_val:]
train_list = train_list[:-n_val]

dataset = {
    "train": train_list,
    "val": val_list,
    "test": test_list
}

with open(output_json, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Created {output_json}")
print(f"Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")