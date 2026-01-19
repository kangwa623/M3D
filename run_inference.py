import torch
import sys
import nibabel as nib
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from monai.transforms import Resize

MODEL = "GoodBaiBai88/M3D-LaMed-Llama-2-7B"

def load_image(path):
    nifti_img = nib.load(path)
    img = nifti_img.get_fdata()
    resize = Resize(spatial_size=(32, 256, 256), mode="bilinear")
    img = resize(img)
    return img.array

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_inference.py <image_path> \"<prompt>\"")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("Loading image...")
    image_np = load_image(image_path)
    image_pt = torch.from_numpy(image_np).unsqueeze(0)
    image_pt = image_pt.to(next(model.parameters()).device)

    full_prompt = "<im_patch>" * 256 + prompt
    input_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(next(model.parameters()).device)

    print("Running inference...")
    with torch.no_grad():
        generation, _ = model.generate(
            image_pt,
            input_ids,
            max_new_tokens=256,
            do_sample=False,
        )

    output = tokenizer.decode(generation[0], skip_special_tokens=True)
    print("\n=== MODEL OUTPUT ===\n")
    print(output)

if __name__ == "__main__":
    main()