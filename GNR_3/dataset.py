import os

import cv2
import torch
from torch.utils.data import Dataset

CAMVID_CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
}


class SegDataset(Dataset):
    def __init__(self, root_dir=None, grayscale=False, image_dir=None, mask_dir=None):
        if image_dir is not None and mask_dir is not None:
            self.image_dir = image_dir
            self.mask_dir = mask_dir
        elif root_dir is not None:
            self.image_dir = os.path.join(root_dir, "images")
            self.mask_dir = os.path.join(root_dir, "masks")
        else:
            raise ValueError("Provide either root_dir or both image_dir and mask_dir.")

        self.grayscale = grayscale
        self.images = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Read image
        if self.grayscale:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
            image = image.astype("float32") / 255.0
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
            image = image.astype("float32") / 255.0

        # Read multiclass mask and keep class ids.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
        remapped_mask = mask.copy()
        for original_id, new_id in CAMVID_CLASS_MAP.items():
            remapped_mask[mask == original_id] = new_id
        mask = remapped_mask.astype("int64")

        # Convert to tensor
        if self.grayscale:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
