import argparse
import os

import cv2
import numpy as np

from segnet import VARIANT_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camvid-root", default="CamVid")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["fcn_basic", "fcn_basic_no_dim_reduction", "bilinear_interpolation"],
        choices=sorted(VARIANT_CONFIGS),
    )
    return parser.parse_args()

def add_title(image, title):
    canvas = np.full((image.shape[0] + 35, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[35:] = image
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def to_three_channel(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def main():
    args = parse_args()
    output_dir = "outputs"

    image_dir = os.path.join(args.camvid_root, "test")
    image_names = sorted(os.listdir(image_dir))

    comparison_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    groups = [
        {"name": "segnet", "variants": ["segnet_basic", "segnet_basic_encoder_addition", "segnet_skip"]},
        {"name": "fcn", "variants": ["fcn_basic", "fcn_basic_no_dim_reduction"]},
        {"name": "bilinear", "variants": ["bilinear_interpolation"]}
    ]

    for index, image_name in enumerate(image_names):
        if index % 50 != 0:
            continue

        original_path = os.path.join(image_dir, image_name)
        original = cv2.imread(original_path)
        original = cv2.resize(original, (128, 128), interpolation=cv2.INTER_LINEAR)

        for group in groups:
            panels = [add_title(original, "Original")]

            for variant in group["variants"]:
                mask_path = os.path.join(output_dir, "predictions", variant, f"pred_color_{index}.png")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Missing prediction for {variant} at {mask_path}. Run test.py --variant {variant} first.")

                mask = cv2.imread(mask_path)
                mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
                panels.append(add_title(mask, variant))

            comparison_image = np.hstack(panels)
            save_path = os.path.join(comparison_dir, f"comparison_{group['name']}_{index}.png")
            cv2.imwrite(save_path, comparison_image)

    print(f"Saved comparison images to {comparison_dir}")


if __name__ == "__main__":
    main()
