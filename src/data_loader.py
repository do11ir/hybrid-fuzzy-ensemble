import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class HeartDiseaseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_heart_data(
    file_path="data/raw/heart.csv",
    train_idx=None,
    val_idx=None,
    batch_size=32
):
    if train_idx is None or val_idx is None:
        raise ValueError("train_idx and val_idx must be provided for cross-validation")

    # Load dataset
    df = pd.read_csv(file_path)

    # Explicit column handling (safer)
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)

    # Index-based split (NO data leakage)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create datasets
    train_dataset = HeartDiseaseDataset(X_train, y_train)
    val_dataset = HeartDiseaseDataset(X_val, y_val)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader
