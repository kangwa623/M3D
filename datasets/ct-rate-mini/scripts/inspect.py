from datasets import load_dataset

ds = load_dataset(
    "ibrahimhamamci/CT-RATE",
    "reports",
    split="train",
    streaming=True
)

for sample in ds:
    print(sample.keys())
    print(sample)
    break