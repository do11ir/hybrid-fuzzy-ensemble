import torch
from src.models import HybridEnsemble
from src.data_loader import load_heart_data
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model_path="results/hybrid_model.pth", data_path="data/raw/heart.csv", input_dim=13, batch_size=16):
    # بارگذاری داده‌ها
    _, val_loader = load_heart_data(data_path, batch_size=batch_size)

    # ساخت مدل و بارگذاری وزن‌ها
    model = HybridEnsemble(input_dim)
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
