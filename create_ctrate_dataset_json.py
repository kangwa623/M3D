# Create this file: create_ctrate_dataset_json.py
import os
import json

# Paths
data_root = "/nfs/usrhome2/africanstu/kangwa/m3d/M3D/ctrate_volumes/m3d_npy"
train_dir = os.path.join(data_root, "train")
test_dir = os.path.join(data_root, "test")
output_json = "/nfs/usrhome2/africanstu/kangwa/m3d/M3D/Data/ctrate_dataset.json"

# Create Data directory if needed
os.makedirs(os.path.dirname(output_json), exist_ok=True)

# Collect training samples
train_list = []
if os.path.exists(train_dir):
    for patient_id in sorted(os.listdir(train_dir)):
        patient_dir = os.path.join(train_dir, patient_id)
        if os.path.isdir(patient_dir):
            img_path = os.path.join(patient_dir, "venous.npy")
            txt_path = os.path.join(patient_dir, f"{patient_id}.txt")
            
            if os.path.exists(img_path) and os.path.exists(txt_path):
                train_list.append({
                    "image": f"train/{patient_id}/venous.npy",
                    "text": f"train/{patient_id}/{patient_id}.txt"
                })

# Collect test samples
test_list = []
if os.path.exists(test_dir):
    for patient_id in sorted(os.listdir(test_dir)):
        patient_dir = os.path.join(test_dir, patient_id)
        if os.path.isdir(patient_dir):
            img_path = os.path.join(patient_dir, "venous.npy")
            txt_path = os.path.join(patient_dir, f"{patient_id}.txt")
            
            if os.path.exists(img_path) and os.path.exists(txt_path):
                test_list.append({
                    "image": f"test/{patient_id}/venous.npy",
                    "text": f"test/{patient_id}/{patient_id}.txt"
                })

# Create dataset structure
dataset = {
    "train": train_list,
    "test": test_list
}

# Save JSON
with open(output_json, 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"Created dataset JSON: {output_json}")
print(f"Training samples: {len(train_list)}")
print(f"Test samples: {len(test_list)}")