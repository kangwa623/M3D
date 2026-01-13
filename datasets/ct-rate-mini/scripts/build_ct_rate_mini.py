# datasets/ct-rate-mini/scripts/build_ct_rate_mini.py

import os
import json
import shutil
from pathlib import Path

# Where CT-RATE is on your server
OFFICIAL_IMG_ROOT = Path(os.path.expanduser("~/kangwa/CT-RATE/dataset/train"))

# Where the mini dataset should be built
OUT_ROOT = Path(__file__).resolve().parents[1] / "data"
IMG_OUT = OUT_ROOT / "images"
ANN_OUT = OUT_ROOT / "annotations.json"

MAX_SAMPLES = 15

def main():
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    collected = []
    for sub in sorted(OFFICIAL_IMG_ROOT.iterdir()):
        if not sub.is_dir():
            continue

        for f in sub.iterdir():
            if f.suffixes[-2:] == [".nii", ".gz"]:
                collected.append(f)
                if len(collected) >= MAX_SAMPLES:
                    break
        if len(collected) >= MAX_SAMPLES:
            break

    if not collected:
        print("No .nii.gz files found under CT-RATE/train/")
        return

    annotations = []
    for i, src in enumerate(collected):
        dst = IMG_OUT / f"case_{i:03d}.nii.gz"
        shutil.copy2(src, dst)
        annotations.append({
            "id": f"case_{i:03d}",
            "source": str(src),
            "image": str(dst)
        })

    with open(ANN_OUT, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Built CT-RATE mini set with {len(collected)} samples.")

if __name__ == "__main__":
    main()