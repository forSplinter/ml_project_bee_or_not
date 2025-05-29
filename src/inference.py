import torch
import os
from torchvision import transforms
from dataloader import BeeNDataset
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from AlexNet import AlexNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/alexnet_init.pt"
BATCH_SIZE = 16
NUM_CLASSES = 3
IMG_DIR = "data/train/img"
MSK_DIR = "data/train/masks"

valid_ids = []
for img_id in range(1, 251):
    msk_path = os.path.join(MSK_DIR, f"binary_{img_id}.tif")
    if os.path.exists(msk_path):
        valid_ids.append(img_id)

IMAGE_IDS = valid_ids


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset = BeeNDataset(
    img_dir=IMG_DIR, msk_dir=MSK_DIR, ids=IMAGE_IDS, transform=transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AlexNet(num_classes=3)
model.load_state_dict((torch.load(MODEL_PATH, map_location=DEVICE)))
model.to(DEVICE)
model.eval()

a_preds = []
a_ids = []

with torch.no_grad():
    for imgs, ids in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        a_preds.extend(preds)
        a_ids.extend(ids)


os.makedirs("outputs", exist_ok=True)
df = pd.DataFrame({"ID": a_ids, "bug_type": a_preds})
df.to_csv("outputs/predictions_alexnet_1.csv", index=False)
print("Saved")
