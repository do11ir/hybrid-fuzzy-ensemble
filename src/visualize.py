import torch
import matplotlib.pyplot as plt
from src.models import HybridEnsemble
from src.data_loader import load_heart_data

def visualize_predictions(model_path="results/hybrid_model.pth", data_path="data/raw/heart.csv", input_dim=13, batch_size=8):
    # بارگذاری داده‌ها
    _, val_loader = load_heart_data(data_path, batch_size=batch_size)

    # ساخت مدل و بارگذاری وزن‌ها
    model = HybridEnsemble(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # گرفتن چند نمونه برای نمایش
    X_batch, y_batch = next(iter(val_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    # خروجی شبکه‌ها
    with torch.no_grad():
        out1 = model.base1(X_batch)
        out2 = model.base2(X_batch)
        voted = model.voting(out1, out2)
        final_out = model.final(voted)
        final_out = torch.sigmoid(final_out)

    # نمایش نتایج
    num_samples = X_batch.size(0)
    indices = range(num_samples)

    plt.figure(figsize=(12,6))
    plt.bar(indices, out1.cpu().numpy().flatten(), width=0.2, label="BaseNN1", align='center')
    plt.bar([i + 0.2 for i in indices], out2.cpu().numpy().flatten(), width=0.2, label="BaseNN2", align='center')
    plt.bar([i + 0.4 for i in indices], voted.cpu().numpy().flatten(), width=0.2, label="Voting Layer", align='center')
    plt.bar([i + 0.6 for i in indices], final_out.cpu().numpy().flatten(), width=0.2, label="Final Output", align='center')

    plt.xticks([i + 0.3 for i in indices], [f"Sample {i+1}" for i in indices])
    plt.ylabel("Output Probability")
    plt.title("Hybrid Fuzzy Ensemble Prediction Flow")
    plt.legend()
    plt.show()

    # نمایش مقدار هدف واقعی
    print("True labels:", y_batch.cpu().numpy().flatten())
