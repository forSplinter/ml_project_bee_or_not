import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class BeeNDataset(Dataset):
    def __init__(self, img_dir, msk_dir, ids, transform=None, labels=None):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.ids = ids  # id for each images in our train
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.JPG")
        msk_path = os.path.join(self.msk_dir, f"binary_{img_id}.tif")

        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")

        img_np = np.array(img)
        msk_np = np.array(msk)
        msk_bool = msk_np > 0
        msked_np = img_np * msk_bool[:, :, None]

        msk_img = Image.fromarray(msked_np)

        if self.transform:
            msk_img = self.transform(msk_img)

        if self.labels:
            label = self.labels[img_id]
            return msk_img, label

        else:
            return msk_img, img_id
