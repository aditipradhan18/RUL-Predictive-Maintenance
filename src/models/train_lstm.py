
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_model import LSTMModel


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "artifacts" / "sequence_data.npz"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "lstm_model.pt"

# Training configuration
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}"
        )

    # Load the prepared arrays
    with np.load(DATA_PATH) as data:
        X_train = data["X_train"].copy()
        y_train = data["y_train"].copy()
        X_val = data["X_val"].copy()
        y_val = data["y_val"].copy()

    expected_shapes = {
        "X_train": (12641, 50, 18),
        "y_train": (12641,),
        "X_val": (3090, 50, 18),
        "y_val": (3090,),
    }

    arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }

    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} has shape {array.shape}; "
                f"expected {expected_shapes[name]}"
            )

        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values")

    # Convert arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    train_dataset = TensorDataset(
        X_train_tensor, y_train_tensor
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device("cpu")
    model = LSTMModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Training sequences: {len(X_train_tensor)}")
    print(f"Validation sequences: {len(X_val_tensor)}")
    print(f"Training for up to {EPOCHS} epochs\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            batch_size = batch_X.size(0)
            total_train_loss += loss.item() * batch_size
            total_samples += batch_size

        train_mse = total_train_loss / total_samples

        # Evaluate on validation data
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor.to(device))
            val_mae = torch.mean(
                torch.abs(val_predictions - y_val_tensor.to(device))
            ).item()
            val_rmse = torch.sqrt(
                torch.mean(
                    (val_predictions - y_val_tensor.to(device)) ** 2
                )
            ).item()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train MSE: {train_mse:.3f} | "
            f"Val MAE: {val_mae:.3f} | "
            f"Val RMSE: {val_rmse:.3f}"
        )

        # Save the model with the lowest validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_size": 18,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "best_val_mae": best_val_mae,
                    "best_val_rmse": val_rmse,
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation MAE: {best_val_mae:.3f} cycles")
    print(f"Best checkpoint saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
