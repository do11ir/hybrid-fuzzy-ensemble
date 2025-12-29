import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.train import train_model
from src.evaluate import evaluate_model

DATA_PATH = "data/raw/heart.csv"
INPUT_DIM = 13
BATCH_SIZE = 8
EPOCHS = 50
LR = 0.001
N_SPLITS = 5


if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    confusion_matrices = []
    reports = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n================ FOLD {fold} / {N_SPLITS} ================")

        model_path = f"results/hybrid_model_fold_{fold}.pth"

        # ---------- TRAIN ----------
        train_model(
            data_path=DATA_PATH,
            train_idx=train_idx,
            val_idx=val_idx,
            input_dim=INPUT_DIM,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            save_path=model_path
        )

        # ---------- EVALUATE (NO LEAKAGE) ----------
        cm, report = evaluate_model(
            model_path=model_path,
            data_path=DATA_PATH,
            train_idx=train_idx,     # REQUIRED
            val_idx=val_idx,         # REQUIRED
            input_dim=INPUT_DIM,
            batch_size=BATCH_SIZE
        )

        confusion_matrices.append(cm)
        reports.append(report)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(pd.DataFrame(report).transpose())

    print("\n================ AVERAGE RESULTS (5-FOLD) ================")

    # Summed confusion matrix (CORRECT)
    sum_cm = np.sum(confusion_matrices, axis=0)
    print("\nSummed Confusion Matrix:")
    print(sum_cm)

    # Average metrics (NO support averaging, NO KeyError)
    metrics = ["precision", "recall", "f1-score"]
    avg_report = {}

    example_report = reports[0]
    class_labels = [k for k in example_report.keys() if k.replace('.', '', 1).isdigit()]
    labels = class_labels + ["macro avg", "weighted avg"]

    for cls in labels:
        avg_report[cls] = {
            m: np.mean([rep[cls][m] for rep in reports])
            for m in metrics
        }

    avg_report = pd.DataFrame(avg_report).T
    print("\nAverage Classification Report (metrics only):")
    print(avg_report)

    print("\n==== CROSS-VALIDATION COMPLETE ====")
