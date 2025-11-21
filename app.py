
import os, random, numpy as np, pandas as pd
from math import sqrt
from tqdm.notebook import tqdm
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def filter_missing_images(df, img_dir):
    valid_rows = []
    missing = 0
    for _, row in df.iterrows():
        img_id = str(row["id"])
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(img_dir, f"{img_id}{ext}")
            if os.path.exists(path):
                row["path"] = path
                valid_rows.append(row)
                break
        else:
            missing += 1
    print(f"✅ Kept {len(valid_rows)} / {len(df)} images (removed {missing} missing files)")
    return pd.DataFrame(valid_rows)
df = pd.read_csv(TRAIN_CSV)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle data

n_val = int(0.1 * len(df))
df_train_raw, df_val_raw = df[:-n_val], df[-n_val:]

df_train = filter_missing_images(df_train_raw, TRAIN_DIR)
df_val   = filter_missing_images(df_val_raw, TRAIN_DIR)

print("Train:", len(df_train), "Val:", len(df_val))
class FaceDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            image = Image.new("RGB", (224,224), (0,0,0))
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row["id"]
        age = torch.tensor(row["age"], dtype=torch.float32)
        gender = torch.tensor(row["gender"], dtype=torch.long)
        return image, age, gender

  train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1,0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


train_dataset = FaceDataset(df_train, transform=train_tf)
val_dataset   = FaceDataset(df_val, transform=val_tf)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
class ScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc_age = nn.Linear(128,1)
        self.fc_gender = nn.Linear(128,2)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        age = self.fc_age(x).squeeze(1)
        gender = self.fc_gender(x)
        return age, gender
class FinetunedResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.fc_age = nn.Linear(in_features, 1)
        self.fc_gender = nn.Linear(in_features, 2)

    def forward(self, x):
        x = self.base(x)
        age = self.fc_age(x).squeeze(1)
        gender = self.fc_gender(x)
        return age, gender
def compute_scores(y_true_age, y_pred_age, y_true_gender, y_pred_gender):
    y_true_age = np.array(y_true_age)
    y_pred_age = np.array(y_pred_age)
    rmse = sqrt(np.mean((y_true_age - y_pred_age)**2))
    age_score = 1 - min(rmse, 30)/30
    f1 = f1_score(y_true_gender, y_pred_gender, average='macro')
    final = 2*(age_score*f1)/(age_score+f1+1e-12)
    return rmse, age_score, f1, final


def train_model(model, name, epochs=5, lr=1e-4):
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_score = -1

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0
        for imgs, ages, genders in tqdm(train_loader, desc=f"[{name}] Epoch {epoch}"):
            imgs, ages, genders = imgs.to(device), ages.to(device), genders.to(device)
            optimizer.zero_grad()
            pred_age, pred_gender = model(imgs)
            loss = mse_loss(pred_age, ages) + ce_loss(pred_gender, genders)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # validation
        model.eval()
        with torch.no_grad():
            yta, ypa, ytg, ypg = [], [], [], []
            for imgs, ages, genders in val_loader:
                imgs = imgs.to(device)
                pred_age, pred_gender = model(imgs)
                yta += ages.tolist()
                ypa += pred_age.cpu().tolist()
                ytg += genders.tolist()
                ypg += torch.argmax(pred_gender,1).cpu().tolist()
            rmse, ascore, f1, final = compute_scores(yta, ypa, ytg, ypg)

        print(f"Epoch {epoch}: Loss={tr_loss:.3f} RMSE={rmse:.3f} F1={f1:.3f} Final={final:.3f}")
        if final > best_score:
            best_score = final
            torch.save(model.state_dict(), f"/kaggle/working/{name}_best.pth")
            print(f"✅ Best model saved ({final:.3f})")

    print(f"Training done. Best final score={best_score:.3f}")
cnn = ScratchCNN().to(device)
train_model(cnn, "scratchcnn_age_gender", epochs=5, lr=1e-3)

resnet = FinetunedResNet(pretrained=True).to(device)
train_model(resnet, "resnet18_age_gender", epochs=5, lr=1e-4)
test_df = pd.read_csv(TEST_CSV)
test_df = filter_missing_images(test_df, TEST_DIR)
test_dataset = FaceDataset(test_df, transform=val_tf, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

model = FinetunedResNet(pretrained=False).to(device)
model.load_state_dict(torch.load("/kaggle/working/resnet18_age_gender_best.pth", map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for imgs, ids in tqdm(test_loader, desc="Inference"):
        imgs = imgs.to(device)
        pred_age, pred_gender = model(imgs)
        ages = pred_age.cpu().numpy()
        genders = torch.argmax(pred_gender, 1).cpu().numpy()
        preds.extend(zip(ids, ages, genders))

submission = pd.DataFrame(preds, columns=["id","age","gender"])
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("submission.csv created successfully!")
submission.head()
