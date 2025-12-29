import torch
import torch.nn as nn
import torch.optim as optim
from src.models import HybridEnsemble
from src.data_loader import load_heart_data
import os
import pandas as pd


def train_model(
    data_path="data/raw/heart.csv",
    train_idx=None,
    val_idx=None,
    input_dim=13,
    epochs=50,
    lr=0.001,
    batch_size=16,
    save_path="results/hybrid_model_fold.pth"
):
    if train_idx is None or val_idx is None:
        raise ValueError("train_idx and val_idx must be provided for cross-validation")

    # Load data for THIS fold
    train_loader, val_loader = load_heart_data(
        file_path=data_path,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=batch_size
    )

    # ---------- CLASS WEIGHT FIX (CRITICAL) ----------
    df = pd.read_csv(data_path)
    y_train = df.iloc[train_idx]["target"].values

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()

    pos_weight = torch.tensor(
        n_neg / n_pos,
        dtype=torch.float32
    )
    # -----------------------------------------------

    # Build model (NEW model per fold)
    model = HybridEnsemble(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ⚠️ FIXED: weighted loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ---------- VALIDATE ----------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    return model, best_val_loss
