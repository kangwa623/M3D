import json
import os

# Load your json file (replace 'data.json' with your actual filename)
with open('m3d_dataset_split.json', 'r') as f:
    data = json.load(f)

# The keys we want to process
splits = ['train', 'val', 'test']

for split in splits:
    if split in data:
        with open(f'{split}_videos.txt', 'w') as out_file:
            for entry in data[split]:
                # Extract '890' from '890/venous.npy'
                file_id = entry['image'].split('/')[0]
                # Format to 'venous_890.mp4'
                formatted_name = f"venous_{file_id}.mp4"
                out_file.write(formatted_name + "\n")
        print(f"Successfully created {split}.txt")