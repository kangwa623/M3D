import os
import json
import shutil
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- Configuration ---
# Local path to your fine-tuned model
model_path = "/nfs/usrhome2/mkfmelbatel/M3D/output/LaMed-Phi3-4B-finetune-0000"
# Path to your dataset split JSON
json_path = "./Data/m3d_dataset_split.json"
# Base directory where your .npy and .txt files are stored
base_data_dir = "/nfs/usrhome2/mkfmelbatel/datasets/trials_report/m3d_npy_v1"
# Output directory
output_base = "M3D_phi3_pred"

device = torch.device('cuda')
dtype = torch.bfloat16
proj_out_num = 256
question = "Can you provide a caption consists of findings for this medical image?"

# --- Setup Directories ---
pred_dir = os.path.join(output_base, "M3D_prediction")
gt_dir = os.path.join(output_base, "ground_truth")
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# --- Load Model and Tokenizer ---
print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)
model.to(device=device)

# --- Load Dataset Split ---
with open(json_path, 'r') as f:
    dataset = json.load(f)

# Use the 'test' split from the JSON
test_samples = dataset.get('test', [])
if not test_samples:
    print("Warning: No 'test' split found in JSON. Checking 'val' as fallback...")
    test_samples = dataset.get('val', [])

# --- Batch Inference Loop ---
print(f"Starting inference on {len(test_samples)} samples...")

for sample in tqdm(test_samples):
    # Paths from JSON
    rel_image_path = sample['image']  # e.g., "890/venous.npy"
    rel_text_path = sample['text']  # e.g., "890/890.txt"

    # Absolute paths
    abs_image_path = os.path.join(base_data_dir, rel_image_path)
    abs_text_path = os.path.join(base_data_dir, rel_text_path)

    # Use the patient ID (folder name) as the filename for the output
    patient_id = rel_text_path.split('/')[0]
    output_filename = f"{patient_id}.txt"

    # 1. Load and Preprocess Image
    try:
        image_np = np.load(abs_image_path)
        image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
    except Exception as e:
        print(f"Error loading image {abs_image_path}: {e}")
        continue

    # 2. Prepare Input Text
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_ids = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

    # 3. Generate Prediction
    with torch.no_grad():
        generation = model.generate(
            image_pt,
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=1.0
        )

    generated_text = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    # Remove the question/prompt from the output if it's included
    final_pred = generated_text.replace(question, "").strip()

    # 4. Save Prediction
    with open(os.path.join(pred_dir, output_filename), 'w', encoding='utf-8') as f_pred:
        f_pred.write(final_pred)

    # 5. Copy Ground Truth
    if os.path.exists(abs_text_path):
        shutil.copy(abs_text_path, os.path.join(gt_dir, output_filename))
    else:
        print(f"Warning: Ground truth file missing for {patient_id}")

print(f"Done! Results saved in '{output_base}' folder.")