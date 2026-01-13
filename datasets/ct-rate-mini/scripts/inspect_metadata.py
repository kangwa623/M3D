from datasets import load_dataset

ds = load_dataset(
    "ibrahimhamamci/CT-RATE",
    "metadata",
    split="train",
    streaming=True
)

for sample in ds:
    print("Keys:", sample.keys())
    print("Sample:", sample)
    break