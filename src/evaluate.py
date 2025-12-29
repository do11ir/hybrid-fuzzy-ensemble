import torch
from src.models import HybridEnsemble
from src.data_loader import load_heart_data
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(
    model_path,
    data_path="data/raw/heart.csv",
    train_idx=None,
    val_idx=None,
    input_dim=13,
    batch_size=16
):
    if train_idx is None or val_idx is None:
        raise ValueError("train_idx and val_idx must be provided for cross-validation")

    # Load validation data using TRAIN scaler (NO leakage)
    _, val_loader = load_heart_data(
        file_path=data_path,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=batch_size
    )

    # Build and load model
    model = HybridEnsemble(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            y_true.extend(y_batch.cpu().numpy().ravel())
            y_pred.extend(preds.cpu().numpy().ravel())

    # Metrics
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    return cm, report
