import torch
from torchviz import make_dot
from src.models import HybridEnsemble
from src.data_loader import load_heart_data

def visualize_model_graph(model_path="results/hybrid_model.pth", input_dim=13):
    # ساخت مدل و بارگذاری وزن‌ها
    model = HybridEnsemble(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # نمونه ورودی
    x = torch.randn(1, input_dim).to(device)

    # محاسبه خروجی
    y = model(x)

    # ساخت گراف
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("hybrid_ensemble_model_graph", format="png")
    print("Graph saved as 'hybrid_ensemble_model_graph.png'")
