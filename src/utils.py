from torchviz import make_dot

def visualize_model(model, input_tensor):
    y = model(input_tensor)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render("hybrid_ensemble_graph", format="png")
    print("Graph saved as hybrid_ensemble_graph.png")
