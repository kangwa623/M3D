import os
import json
import shutil
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- Configuration ---
# Local path to your fine-tuned HF-style model (must contain config.json, tokenizer, weights)
model_path = "/home/africanstu/kangwa/m3d/M3D/output/LaMed-Phi3-4B-finetune-0000"

# Base directory where your .npy and .txt files are stored
base_data_dir = "/home/africanstu/kangwa/m3d/M3D/datasets/ct-rate-mini/m3d_npy"

# Output directory
output_base = "M3D_phi3_pred"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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
    device_map="auto",
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
model.eval()

# --- Enumerate Dataset (ct-rate-mini style) ---
cases = []
for d in sorted(os.listdir(base_data_dir)):
    case_dir = os.path.join(base_data_dir, d)
    if not os.path.isdir(case_dir):
        continue
    img = os.path.join(case_dir, "venous.npy")
    txt = os.path.join(case_dir, f"{d}.txt")
    if os.path.exists(img) and os.path.exists(txt):
        cases.append((d, img, txt))

print(f"Starting inference on {len(cases)} samples...")

# --- Batch Inference Loop ---
for patient_id, abs_image_path, abs_text_path in tqdm(cases):

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
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device=device)

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
    final_pred = generated_text.replace(question, "").strip()

    # 4. Save Prediction
    with open(os.path.join(pred_dir, output_filename), "w", encoding="utf-8") as f_pred:
        f_pred.write(final_pred)

    # 5. Copy Ground Truth
    if os.path.exists(abs_text_path):
        shutil.copy(abs_text_path, os.path.join(gt_dir, output_filename))
    else:
        print(f"Warning: Ground truth file missing for {patient_id}")

print(f"Done! Results saved in '{output_base}' folder.")