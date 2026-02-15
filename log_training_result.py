#!/usr/bin/env python3
"""
Append training run summary to a results log (CSV + optional markdown).
Usage:
  python log_training_result.py
  python log_training_result.py --log path/to/training.log
  python log_training_result.py --log path/to/training.log --output_dir path/to/LaMed/output/LaMed-Phi3-4B-pretrain
"""

import argparse
import os
import re
from datetime import datetime

# Default paths relative to project root (script's parent directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG = os.path.join(SCRIPT_DIR, "LaMed", "output", "LaMed-Phi3-4B-pretrain", "training.log")
RESULTS_CSV = os.path.join(SCRIPT_DIR, "training_results.csv")
RESULTS_MD = os.path.join(SCRIPT_DIR, "training_results.md")


def parse_training_log(log_path):
    """Parse final summary line and optional per-step losses from training.log."""
    if not os.path.isfile(log_path):
        return None, []
    with open(log_path, "r") as f:
        lines = f.readlines()
    # Final summary: {'train_runtime': 137.74, 'train_samples_per_second': 3.006, ...}
    summary = None
    step_losses = []
    for line in lines:
        if "'train_runtime'" in line or '"train_runtime"' in line:
            # Extract key=value pairs
            summary = {}
            for match in re.finditer(r"['\"]?(\w+)['\"]?\s*:\s*([\d.e+-]+)", line):
                key, val = match.group(1), match.group(2)
                try:
                    summary[key] = float(val)
                except ValueError:
                    summary[key] = val
        # Per-step loss lines: {'loss': 2.4796, ...}
        m = re.search(r"['\"]?loss['\"]?\s*:\s*([\d.]+)", line)
        if m and "train_runtime" not in line:
            step_losses.append(float(m.group(1)))
    return summary, step_losses


def ensure_csv_header(csv_path):
    if not os.path.isfile(csv_path):
        with open(csv_path, "w") as f:
            f.write("timestamp,output_dir,run_dir,train_loss,train_runtime_s,train_samples_per_second,train_steps_per_second,epochs,steps\n")


def append_csv_row(csv_path, row_dict):
    ensure_csv_header(csv_path)
    with open(csv_path, "a") as f:
        line = ",".join(str(row_dict.get(k, "")) for k in [
            "timestamp", "output_dir", "run_dir", "train_loss", "train_runtime_s",
            "train_samples_per_second", "train_steps_per_second", "epochs", "steps"
        ])
        f.write(line + "\n")


def append_md_table(md_path, row_dict):
    """Append one row to a markdown table (create file with header if needed)."""
    headers = ["Timestamp", "Output dir", "Train loss", "Runtime (s)", "Samples/s", "Steps/s", "Epochs", "Steps"]
    if not os.path.isfile(md_path):
        with open(md_path, "w") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
    with open(md_path, "a") as f:
        row = (
            row_dict.get("timestamp", ""),
            row_dict.get("output_dir", ""),
            row_dict.get("train_loss", ""),
            row_dict.get("train_runtime_s", ""),
            row_dict.get("train_samples_per_second", ""),
            row_dict.get("train_steps_per_second", ""),
            row_dict.get("epochs", ""),
            row_dict.get("steps", ""),
        )
        f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Log training run to CSV and optional markdown.")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Path to training.log")
    parser.add_argument("--output_dir", default=None, help="Output directory for this run (default: inferred from --log)")
    parser.add_argument("--csv", default=RESULTS_CSV, help="Path to results CSV")
    parser.add_argument("--md", default=RESULTS_MD, help="Path to results markdown table")
    parser.add_argument("--no_md", action="store_true", help="Do not write markdown table")
    args = parser.parse_args()

    summary, _ = parse_training_log(args.log)
    if not summary:
        print(f"No summary found in {args.log}; skipping.")
        return

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.log)
    run_dir = os.path.basename(output_dir.rstrip("/"))

    row = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": output_dir,
        "run_dir": run_dir,
        "train_loss": summary.get("train_loss", ""),
        "train_runtime_s": summary.get("train_runtime", ""),
        "train_samples_per_second": summary.get("train_samples_per_second", ""),
        "train_steps_per_second": summary.get("train_steps_per_second", ""),
        "epochs": summary.get("epoch", ""),
        "steps": int(summary.get("train_steps", summary.get("steps", 0)) or 0),
    }
    # Infer steps if present in log
    if row["steps"] == 0 and "train_runtime" in summary:
        rt = summary.get("train_runtime") or 0
        sps = summary.get("train_steps_per_second") or 0
        if sps:
            row["steps"] = int(round(rt * sps))

    append_csv_row(args.csv, row)
    if not args.no_md:
        append_md_table(args.md, row)
    print(f"Appended result to {args.csv}" + (f" and {args.md}" if not args.no_md else ""))


if __name__ == "__main__":
    main()