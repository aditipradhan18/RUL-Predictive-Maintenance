
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.lstm_model import LSTMModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "lstm_model.pt"

app = FastAPI(
    title="RUL Prediction API",
    description="Estimate remaining useful life from simulated turbofan sensor sequences.",
    version="1.0.0",
)

model = None
device = torch.device("cpu")


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"LSTM checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = LSTMModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="50 time steps, each containing 18 features in training order.",
    )


@app.get("/")
def home():
    return {
        "message": "RUL Prediction API is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    sequence = np.asarray(request.sequence, dtype=np.float32)

    if sequence.shape != (50, 18):
        raise HTTPException(
            status_code=422,
            detail="sequence must contain exactly 50 time steps and 18 features per step",
        )

    if not np.isfinite(sequence).all():
        raise HTTPException(
            status_code=422,
            detail="sequence must contain only finite numbers",
        )

    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor).item()

    return {
        "predicted_rul_cycles": round(float(prediction), 2),
        "unit": "cycles",
    }
