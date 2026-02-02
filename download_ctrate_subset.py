import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download # Use this for easy, secure downloads

# The library automatically looks for the 'HF_TOKEN' environment variable
ds = load_dataset("ibrahimhamamci/CT-RATE", "reports", split="train", streaming=True)

for item in ds:
    # Use hf_hub_download instead of raw requests. 
    # It handles auth, caching, and retries automatically!
    file_path = hf_hub_download(
        repo_id="ibrahimhamamci/CT-RATE",
        filename=f"dataset/train/{item['VolumeName']}",
        repo_type="dataset"
    )
    print(f"Downloaded to: {file_path}")
    break