import argparse
import os
import re
import sys
import bleach
import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import nibabel as nib
from PIL import Image
from monai.transforms import Resize
from LaMed.src.model.language_model import *

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args(args):
    parser = argparse.ArgumentParser(description="M3D-LaMed chat")
    parser.add_argument('--model_name_or_path', type=str, default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/hf/", choices=[])
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--seg_enable', type=bool, default=True)
    parser.add_argument('--proj_out_num', type=int, default=256)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    # NEW: CLI inference arguments
    parser.add_argument("--image", type=str, default=None, help="Path to image (.nii.gz, .npy, .png, .jpg)")
    parser.add_argument("--question", type=str, default=None, help="Text prompt/question")

    return parser.parse_args(args)

# -----------------------------
# Image loading
# -----------------------------
def image_process(file_path):
    if file_path.endswith('.nii.gz'):
        nifti_img = nib.load(file_path)
        img_array = nifti_img.get_fdata()
    elif file_path.endswith(('.png', '.jpg', '.bmp')):
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        img_array = img_array[np.newaxis, :, :]
    elif file_path.endswith('.npy'):
        img_array = np.load(file_path)
    else:
        raise ValueError("Unsupported file type")

    resize = Resize(spatial_size=(32, 256, 256), mode="bilinear")
    img_meta = resize(img_array)
    img_array, img_affine = img_meta.array, img_meta.affine

    return img_array, img_affine

args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# -----------------------------
# Model setup
# -----------------------------
device = torch.device(args.device)

dtype = torch.float32
if args.precision == "bf16":
    dtype = torch.bfloat16
elif args.precision == "fp16":
    dtype = torch.half

kwargs = {"torch_dtype": dtype}
if args.load_in_4bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        }
    )
elif args.load_in_8bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        }
    )

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    model_max_length=args.max_length,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    device_map='auto',
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    **kwargs
)
model = model.to(device=device)
model.eval()

# -----------------------------
# Core inference
# -----------------------------
def run_single_inference(image_path, question):
    image_np, _ = image_process(image_path)
    prompt = "<im_patch>" * args.proj_out_num + bleach.clean(question)

    input_id = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device=device)
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

    generation, _ = model.generate(
        image_pt,
        input_id,
        seg_enable=args.seg_enable,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    return output_str

# -----------------------------
# CLI MODE
# -----------------------------
if args.image and args.question:
    print("Running in CLI inference mode...")
    print("Image:", args.image)
    print("Question:", args.question)
    answer = run_single_inference(args.image, args.question)
    print("\n=== MODEL OUTPUT ===")
    print(answer)
    sys.exit(0)

# -----------------------------
# Otherwise: launch Gradio UI
# (your original Gradio code follows unchanged)
# -----------------------------
# [Gradio part remains exactly as in your original file]


