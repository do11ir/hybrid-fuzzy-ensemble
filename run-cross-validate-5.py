import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    confusion_matrices = []
    reports = []
    fold_metrics = []

    all_train_losses = []
    all_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n================ FOLD {fold} / {N_SPLITS} ================")

        model_path = f"results/hybrid_model_fold_{fold}.pth"

        # ---------- TRAIN ----------
        model, best_val_loss, train_losses, val_losses = train_model(
            data_path=DATA_PATH,
            train_idx=train_idx,
            val_idx=val_idx,
            input_dim=INPUT_DIM,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            save_path=model_path
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # ---------- EVALUATE ----------
        cm, report = evaluate_model(
            model_path=model_path,
            data_path=DATA_PATH,
            train_idx=train_idx,
            val_idx=val_idx,
            input_dim=INPUT_DIM,
            batch_size=BATCH_SIZE
        )

        confusion_matrices.append(cm)
        reports.append(report)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(pd.DataFrame(report).transpose())

        # ---------- جمع‌آوری متریک‌ها ----------
        accuracy = np.trace(cm) / np.sum(cm)
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        f1 = report["macro avg"]["f1-score"]

        fold_metrics.append({
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    # ---------- میانگین نتایج ----------
    print("\n================ AVERAGE RESULTS (5-FOLD) ================")
    sum_cm = np.sum(confusion_matrices, axis=0)
    print("\nSummed Confusion Matrix:")
    print(sum_cm)

    metrics = ["precision", "recall", "f1-score"]
    avg_report = {}
    example_report = reports[0]
    class_labels = [k for k in example_report.keys() if k.replace('.', '', 1).isdigit()]
    labels = class_labels + ["macro avg", "weighted avg"]

    for cls in labels:
        avg_report[cls] = {m: np.mean([rep[cls][m] for rep in reports]) for m in metrics}

    avg_report = pd.DataFrame(avg_report).T
    print("\nAverage Classification Report (metrics only):")
    print(avg_report)

    # ---------- نمودار grouped bar متریک‌ها ----------
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    folds = metrics_df["fold"].values
    x = np.arange(len(folds))
    width = 0.2

    plt.figure(figsize=(12,6))
    for i, metric in enumerate(metrics_names):
        plt.bar(x + i*width - 1.5*width, metrics_df[metric], width=width, label=metric)

    plt.xticks(x, [f"Fold {f}" for f in folds])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xlabel("Fold")
    plt.title("Comparison of Metrics Across 5-Folds")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/fold_metrics_grouped.png")
    plt.show()

    # ---------- نمودار loss هر fold ----------
    epochs_range = np.arange(1, EPOCHS+1)
    for i in range(N_SPLITS):
        plt.figure(figsize=(10,5))
        plt.plot(epochs_range, all_train_losses[i], '--', label='Train Loss')
        plt.plot(epochs_range, all_val_losses[i], '-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {i+1} Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/loss_fold_{i+1}.png")
        plt.show()

    # ---------- نمودار میانگین loss ----------
    mean_train = np.mean(all_train_losses, axis=0)
    mean_val = np.mean(all_val_losses, axis=0)

    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, mean_train, '--', label='Mean Train Loss')
    plt.plot(epochs_range, mean_val, '-', label='Mean Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Loss Across 5 Folds')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/loss_mean_5fold.png")
    plt.show()

    print("\n==== CROSS-VALIDATION COMPLETE ====")
