import argparse
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SegDataset
from metrics import metric_dict
from segnet import NUM_CLASSES, VARIANT_CONFIGS, build_model

CLASS_COLORS = np.array(
    [
        [128, 128, 128],
        [128, 0, 0],
        [192, 192, 128],
        [128, 64, 128],
        [0, 0, 192],
        [128, 128, 0],
        [192, 128, 128],
        [64, 64, 128],
        [64, 0, 128],
        [64, 64, 0],
        [0, 128, 192],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # Simple test options for choosing variant and dataset path.
    parser.add_argument("--variant", default="segnet_skip", choices=sorted(VARIANT_CONFIGS))
    parser.add_argument("--camvid-root", default="CamVid")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()

def colorize_mask(mask):
    return CLASS_COLORS[mask]


def main():
    args = parse_args()
    # Use CPU for testing.
    device = torch.device("cpu")
    output_dir = "outputs"

    checkpoint_path = os.path.join(output_dir, "checkpoints", args.variant, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py --variant {args.variant} first.")

    grayscale = VARIANT_CONFIGS[args.variant]["input_channels"] == 1
    test_dataset = SegDataset(
        grayscale=grayscale,
        image_dir=os.path.join(args.camvid_root, "test"),
        mask_dir=os.path.join(args.camvid_root, "testannot"),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the trained checkpoint for the selected variant.
    model = build_model(args.variant).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    total_params = sum(param.numel() for param in model.parameters())

    predictions_dir = os.path.join(output_dir, "predictions", args.variant)
    os.makedirs(predictions_dir, exist_ok=True)

    # Store totals first, then average them at the end.
    totals = {"G": 0.0, "C": 0.0, "mIoU": 0.0, "IoU": 0.0, "Dice": 0.0}

    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image = image.to(device)
            mask = mask.to(device)

            pred = model(image)
            batch_metrics = metric_dict(pred, mask, NUM_CLASSES)
            for key, value in batch_metrics.items():
                totals[key] += value

            # Save predicted class id mask.
            pred_np = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(predictions_dir, f"pred_{i}.png"), pred_np)

            # Save a color version of the predicted mask for easier viewing.
            color_pred = colorize_mask(pred_np)
            cv2.imwrite(os.path.join(predictions_dir, f"pred_color_{i}.png"), cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))

    averages = {key: value / len(test_loader) for key, value in totals.items()}
    results = {
        "variant": args.variant,
        "total_parameters": total_params,
        "G": averages["G"],
        "C": averages["C"],
        "mIoU": averages["mIoU"],
        "IoU": averages["IoU"],
        "Dice": averages["Dice"],
    }

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{args.variant}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nTest Results for {args.variant}:")
    print(f"Parameters: {results['total_parameters']}")
    print(f"G: {averages['G']:.4f}")
    print(f"C: {averages['C']:.4f}")
    print(f"mIoU: {averages['mIoU']:.4f}")
    print(f"IoU: {averages['IoU']:.4f}")
    print(f"Dice: {averages['Dice']:.4f}")


if __name__ == "__main__":
    main()
