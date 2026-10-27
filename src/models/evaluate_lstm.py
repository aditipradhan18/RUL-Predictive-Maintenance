
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.lstm_model import LSTMModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "artifacts" / "sequence_data.npz"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "lstm_model.pt"

BATCH_SIZE = 128
DEVICE = torch.device("cpu")


def main():
    # Check required files
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Validation data not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    # Load validation data
    data = np.load(DATA_PATH)

    X_val = data["X_val"]
    y_val = data["y_val"]

    if X_val.ndim != 3 or X_val.shape[1:] != (50, 18):
        raise ValueError(f"Unexpected X_val shape: {X_val.shape}")

    if y_val.shape != (len(X_val),):
        raise ValueError(
            f"Unexpected y_val shape: {y_val.shape}; "
            f"expected ({len(X_val)},)"
        )

    if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
        raise ValueError("Validation data contains NaN or infinite values.")

    # Load the saved LSTM checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = LSTMModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate predictions for all validation samples
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X_val), BATCH_SIZE):
            batch = torch.from_numpy(
                X_val[start:start + BATCH_SIZE].astype(np.float32)
            ).to(DEVICE)

            batch_predictions = model(batch)
            predictions.extend(batch_predictions.cpu().numpy().tolist())

    predictions = np.asarray(predictions, dtype=np.float64)
    actual = y_val.astype(np.float64)

    # Overall evaluation metrics
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    print("\n=== LSTM Validation Evaluation ===")
    print(f"Validation samples: {len(actual)}")
    print(f"Best training epoch: {checkpoint.get('epoch', 'not recorded')}")
    print(
        "Recorded best validation MAE: "
        f"{checkpoint.get('val_mae', 'not recorded')}"
    )
    print(f"Recalculated validation MAE: {mae:.4f} cycles")
    print(f"Recalculated validation RMSE: {rmse:.4f} cycles")
    print(f"Mean actual RUL: {actual.mean():.2f} cycles")
    print(f"Mean predicted RUL: {predictions.mean():.2f} cycles")

    # Prediction error analysis
    errors = predictions - actual
    absolute_errors = np.abs(errors)

    print("\n=== Prediction Error Analysis ===")
    print(f"Mean signed error: {errors.mean():.2f} cycles")
    print(f"Mean absolute error: {absolute_errors.mean():.2f} cycles")
    print(f"Maximum absolute error: {absolute_errors.max():.2f} cycles")
    print(f"90th percentile absolute error: {np.percentile(absolute_errors, 90):.2f} cycles")

    # Error breakdown by actual RUL range
    ranges = [
        ("0-50 cycles", 0, 50),
        ("50-100 cycles", 50, 100),
        ("100-150 cycles", 100, 150),
        ("150+ cycles", 150, float("inf")),
    ]

    print("\n=== Error by Actual RUL Range ===")

    for label, lower, upper in ranges:
        mask = (actual >= lower) & (actual < upper)

        if mask.any():
            range_mae = mean_absolute_error(
                actual[mask],
                predictions[mask],
            )
            range_rmse = np.sqrt(
                mean_squared_error(
                    actual[mask],
                    predictions[mask],
                )
            )

            print(
                f"{label}: "
                f"samples={int(mask.sum())}, "
                f"MAE={range_mae:.2f}, "
                f"RMSE={range_rmse:.2f} cycles"
            )
        else:
            print(f"{label}: no validation samples")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
