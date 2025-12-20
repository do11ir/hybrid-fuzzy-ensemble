from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import visualize_predictions
from src.visualize_graph import visualize_model_graph
from src.visualize_graph_real import visualize_real_flow

DATA_PATH = "data/raw/heart.csv"
MODEL_PATH = "results/hybrid_model.pth"
INPUT_DIM = 13
BATCH_SIZE = 8
EPOCHS = 50
LR = 0.001

if __name__ == "__main__":
    print("==== Training Hybrid Fuzzy Ensemble ====")
    model = train_model(
        data_path=DATA_PATH,
        input_dim=INPUT_DIM,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        save_path=MODEL_PATH
    )

    print("\n==== Evaluating Model ====")
    evaluate_model(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        input_dim=INPUT_DIM,
        batch_size=BATCH_SIZE
    )

    print("\n==== Visualizing Sample Predictions ====")
    visualize_predictions(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        input_dim=INPUT_DIM,
        batch_size=BATCH_SIZE
    )

    print("\n==== Visualizing Model Graph ====")
    visualize_model_graph(
        model_path=MODEL_PATH,
        input_dim=INPUT_DIM
    )

    print("\n==== Visualizing Real Validation Samples with Confidence ====")
    visualize_real_flow(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        input_dim=INPUT_DIM,
        batch_size=5
    )
