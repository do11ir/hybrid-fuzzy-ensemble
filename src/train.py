import torch
import torch.nn as nn
import torch.optim as optim
from src.models import HybridEnsemble
from src.data_loader import load_heart_data
import pandas as pd
import os

def train_model(
    data_path="data/raw/heart.csv",
    input_dim=13,
    epochs=50,
    lr=0.001,
    batch_size=16,
    save_path="results/hybrid_model.pth"
):
    # بارگذاری داده‌ها
    train_loader, val_loader = load_heart_data(data_path, batch_size=batch_size)

    # ساخت مدل
    model = HybridEnsemble(input_dim)

    # انتخاب device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss و Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ارزیابی روی validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                predicted = (outputs >= 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ذخیره بهترین مدل
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print("Training complete. Best model saved at:", save_path)
    return model
