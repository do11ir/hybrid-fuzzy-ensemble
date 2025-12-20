import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HeartDiseaseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_heart_data(file_path="data/raw/heart.csv", test_size=0.2, random_state=42, batch_size=32):
    # بارگذاری دیتاست
    df = pd.read_csv(file_path)

    # متغیر هدف باینری (در این نسخه target از قبل باینری است، پس نیازی به تبدیل نیست)
    y = df['target']
    X = df.drop(columns=['target'])

    # استاندارد سازی ویژگی‌ها
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # تقسیم داده‌ها
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ساخت PyTorch Dataset
    train_dataset = HeartDiseaseDataset(X_train, y_train.values)
    val_dataset = HeartDiseaseDataset(X_val, y_val.values)

    # ساخت DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
