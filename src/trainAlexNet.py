import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataloader import BeeNDataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
from AlexNet import AlexNet
from sklearn.preprocessing import LabelEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
IMG_DIR = "data/train/img"
MSK_DIR = "data/train/masks"
LABEL_FILE = "data/label.xlsx"

df = pd.read_excel(LABEL_FILE)

top_2 = df["bug type"].value_counts().head(2).index.tolist()
df = df[df["bug type"].isin(top_2)].copy()

le = LabelEncoder()
df["bug_type_encoded"] = le.fit_transform(df["bug type"])

label_dict = dict(zip(df["ID"], df["bug_type_encoded"]))

valid_ids = []
for img_id in range(1, 251):
    msk_path = os.path.join(MSK_DIR, f"binary_{img_id}.tif")
    if os.path.exists(msk_path) and img_id in label_dict:
        valid_ids.append(img_id)

IMAGE_IDS = valid_ids
NUM_CLASSES = len(set(label_dict[img_id] for img_id in IMAGE_IDS))

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset = BeeNDataset(
    img_dir=IMG_DIR,
    msk_dir=MSK_DIR,
    ids=IMAGE_IDS,
    transform=transform,
    labels=label_dict,
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AlexNet(num_classes=NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"[{epoch + 1}/{EPOCHS}] Loss: {running_loss:.4f} | Accuracy: {acc:.4f}")

torch.save(model.state_dict(), "models/alexnet_finetuned.pt")
print("model saved")
