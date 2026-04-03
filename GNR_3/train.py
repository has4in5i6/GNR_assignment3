import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SegDataset
from loss import combined_loss
from metrics import mean_iou
from segnet import NUM_CLASSES, VARIANT_CONFIGS, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="segnet_skip", choices=sorted(VARIANT_CONFIGS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camvid-root", default="CamVid")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(dataset, num_classes):
    # Compute simple median frequency balancing weights from the training masks.
    class_counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, mask in dataset:
        counts = torch.bincount(mask.view(-1), minlength=num_classes).float()
        class_counts += counts

    class_frequencies = class_counts / class_counts.sum()
    valid = class_frequencies > 0
    median_frequency = class_frequencies[valid].median()

    weights = torch.ones(num_classes, dtype=torch.float32)
    weights[valid] = median_frequency / class_frequencies[valid]
    return weights

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    output_dir = "outputs"
    checkpoint_dir = os.path.join(output_dir, "checkpoints", args.variant)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    grayscale = VARIANT_CONFIGS[args.variant]["input_channels"] == 1

    # Using the original CamVid train and val folders directly.
    train_dataset = SegDataset(
        grayscale=grayscale,
        image_dir=os.path.join(args.camvid_root, "train"),
        mask_dir=os.path.join(args.camvid_root, "trainannot"),
    )
    val_dataset = SegDataset(
        grayscale=grayscale,
        image_dir=os.path.join(args.camvid_root, "val"),
        mask_dir=os.path.join(args.camvid_root, "valannot"),
    )

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Build the selected model variant.
    model = build_model(args.variant).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES).to(device)

    best_miou = 0.0

    print(f"Training variant: {args.variant}")
    print(VARIANT_CONFIGS[args.variant]["description"])

    for epoch in range(args.epochs):
        # Training step
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = combined_loss(preds, masks, NUM_CLASSES, class_weights=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation step
        model.eval()
        val_miou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                val_miou += mean_iou(preds, masks, NUM_CLASSES)

        avg_val_miou = val_miou / len(val_loader)

        print(f"Epoch [{epoch + 1}/{args.epochs}] Train Loss: {avg_train_loss:.4f} Val mIoU: {avg_val_miou:.4f}")

        # Save the best checkpoint based on validation mIoU.
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

    print("Training complete")


if __name__ == "__main__":
    main()
