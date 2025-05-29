from PIL import Image
import numpy as np


def load_data(img_id):
    img = Image.open(f"train/img/{img_id}.JPG").convert("RGB")
    msk = Image.open(f"train/masks/binary_{img_id}.JPG").convert("L")

    img_np = np.array(img)
    msk_np = np.array(msk)
