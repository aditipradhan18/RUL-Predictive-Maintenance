
from pathlib import Path

import mlflow
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "lstm_model.pt"

mlflow.set_experiment("RUL-LSTM-Experiments")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    with mlflow.start_run(run_name="LSTM-Best-Checkpoint"):
        mlflow.log_param("model", "LSTM")
        mlflow.log_param("input_size", checkpoint["input_size"])
        mlflow.log_param("hidden_size", checkpoint["hidden_size"])
        mlflow.log_param("num_layers", checkpoint["num_layers"])
        mlflow.log_param("dropout", checkpoint["dropout"])
        mlflow.log_param("best_epoch", checkpoint["epoch"])

        mlflow.log_metric(
            "validation_mae",
            checkpoint["best_val_mae"],
        )
        mlflow.log_metric(
            "validation_rmse",
            checkpoint["best_val_rmse"],
        )

        mlflow.log_artifact(str(MODEL_PATH))

        print("MLflow run logged successfully!")
        print(f"Best epoch: {checkpoint['epoch']}")
        print(
            f"Validation MAE: "
            f"{checkpoint['best_val_mae']:.3f} cycles"
        )
        print(
            f"Validation RMSE: "
            f"{checkpoint['best_val_rmse']:.3f} cycles"
        )
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
