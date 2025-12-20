import torch
from src.models import HybridEnsemble
from src.data_loader import load_heart_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_real_flow(model_path="results/hybrid_model.pth", data_path="data/raw/heart.csv", input_dim=13, batch_size=5):
    # بارگذاری داده‌ها
    _, val_loader = load_heart_data(data_path, batch_size=batch_size)

    # ساخت مدل و بارگذاری وزن‌ها
    model = HybridEnsemble(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # گرفتن یک batch از validation
    X_batch, y_batch = next(iter(val_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    with torch.no_grad():
        out1 = model.base1(X_batch)
        out2 = model.base2(X_batch)
        voted = model.voting(out1, out2)
        final_out = torch.sigmoid(model.final(voted))

    # محاسبه confidence برای رنگ‌بندی
    conf = torch.abs(final_out - 0.5) * 2  # فاصله از 0.5 → اعتماد بیشتر
    colors = conf.cpu().numpy().flatten()

    num_samples = X_batch.size(0)
    indices = range(num_samples)

    # ایجاد figure و axis مشخص
    fig, ax = plt.subplots(figsize=(12,6))

    ax.bar(indices, out1.cpu().numpy().flatten(), width=0.2, label="BaseNN1", align='center', color='skyblue')
    ax.bar([i + 0.2 for i in indices], out2.cpu().numpy().flatten(), width=0.2, label="BaseNN2", align='center', color='orange')
    ax.bar([i + 0.4 for i in indices], voted.cpu().numpy().flatten(), width=0.2, label="Voting Layer", align='center', color='green')

    # رنگ‌بندی final output بر اساس confidence
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=1)
    for i, val in enumerate(final_out.cpu().numpy().flatten()):
        ax.bar(i + 0.6, val, width=0.2, color=cmap(norm(colors[i])))

    ax.set_xticks([i + 0.3 for i in indices])
    ax.set_xticklabels([f"Sample {i+1}" for i in indices])
    ax.set_ylabel("Output Probability")
    ax.set_title("Hybrid Fuzzy Ensemble Prediction Flow (Real Validation Samples)")

    # اضافه کردن colorbar به axis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Confidence')

    ax.legend()
    plt.show()

    print("True labels:", y_batch.cpu().numpy().flatten())
    print("Predicted labels:", (final_out >= 0.5).float().cpu().numpy().flatten())
    print("Confidence:", colors)
