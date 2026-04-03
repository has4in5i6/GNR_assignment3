import argparse
import csv
import json
import os
import subprocess
import sys

from segnet import VARIANT_CONFIGS, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_CONFIGS.keys()), choices=sorted(VARIANT_CONFIGS))
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = "outputs"
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    summary_json_path = os.path.join(metrics_dir, "variant_summary.json")
    summary_csv_path = os.path.join(metrics_dir, "variant_summary.csv")
    rows = []

    for variant in args.variants:
        print("=" * 60)
        print(f"Variant: {variant}")
        print(VARIANT_CONFIGS[variant]["description"])
        print("=" * 60)
        subprocess.run([sys.executable, "test.py", "--variant", variant], check=True)
        metrics_path = os.path.join(metrics_dir, f"{variant}.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            row = json.load(f)
        if "total_parameters" not in row:
            row["total_parameters"] = sum(param.numel() for param in build_model(variant).parameters())
        rows.append(row)
        print()

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "total_parameters", "G", "C", "mIoU", "IoU", "Dice"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON summary to {summary_json_path}")
    print(f"Saved CSV summary to {summary_csv_path}")


if __name__ == "__main__":
    main()
