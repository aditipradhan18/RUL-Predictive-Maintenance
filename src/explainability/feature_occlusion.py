
"""Feature occlusion analysis for the trained RUL LSTM."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.data_ingestion import FEATURE_COLUMNS
from src.models.lstm_model import LSTMModel


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "sequence_data.npz"
CHECKPOINT_PATH = ARTIFACTS_DIR / "lstm_model.pt"


def load_model():
    """Load the trained LSTM checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = LSTMModel(
        input_size=checkpoint.get("input_size", 18),
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.2),
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict(model, sequence):
    """Predict RUL from a single 50-by-18 input sequence."""
    sequence = np.asarray(sequence, dtype=np.float32)

    expected_shape = (50, len(FEATURE_COLUMNS))
    if sequence.shape != expected_shape:
        raise ValueError(
            f"Expected sequence shape {expected_shape}, "
            f"received {sequence.shape}."
        )

    if not np.isfinite(sequence).all():
        raise ValueError("Sequence contains non-finite values.")

    tensor = torch.from_numpy(sequence).unsqueeze(0)

    with torch.inference_mode():
        prediction = model(tensor).item()

    if not np.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction.")

    return float(prediction)


def explain_feature_occlusion(
    model,
    sequence,
    feature_baseline,
):
    """
    Measure prediction sensitivity to replacing one feature at a time.

    Each feature is replaced across all time steps with its validation
    median. The original cycle trajectory is preserved.

    This measures model sensitivity, not causation.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    feature_baseline = np.asarray(
        feature_baseline, dtype=np.float32
    )

    expected_shape = (50, len(FEATURE_COLUMNS))

    if sequence.shape != expected_shape:
        raise ValueError(
            f"Expected sequence shape {expected_shape}, "
            f"received {sequence.shape}."
        )

    if feature_baseline.shape != (len(FEATURE_COLUMNS),):
        raise ValueError(
            "Feature baseline must contain one value per feature."
        )

    if not np.isfinite(sequence).all():
        raise ValueError("Sequence contains non-finite values.")

    if not np.isfinite(feature_baseline).all():
        raise ValueError("Feature baseline contains non-finite values.")

    original_prediction = predict(model, sequence)
    results = []

    for index, feature_name in enumerate(FEATURE_COLUMNS):
        # Preserve the actual cycle trajectory. It is unscaled and
        # has a meaningful temporal progression.
        if feature_name == "cycle":
            continue

        occluded = sequence.copy()
        occluded[:, index] = feature_baseline[index]

        occluded_prediction = predict(model, occluded)
        signed_change = occluded_prediction - original_prediction

        results.append(
            {
                "feature": feature_name,
                "original_prediction": original_prediction,
                "occluded_prediction": occluded_prediction,
                "signed_change_cycles": signed_change,
                "importance_cycles": abs(signed_change),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("importance_cycles", ascending=False)
        .reset_index(drop=True)
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sequence data not found: {DATA_PATH}"
        )

    with np.load(DATA_PATH) as data:
        X_val = data["X_val"]
        y_val = data["y_val"]

    if len(X_val) == 0:
        raise ValueError("Validation dataset is empty.")

    model = load_model()

    # Calculate a representative baseline for each feature using
    # the validation sequences. Cycle is excluded from occlusion.
    feature_baseline = np.median(
        X_val.reshape(-1, X_val.shape[-1]),
        axis=0,
    ).astype(np.float32)

    sequence_index = 0
    sequence = X_val[sequence_index]
    actual_rul = float(y_val[sequence_index])

    original_prediction = predict(model, sequence)
    explanation = explain_feature_occlusion(
        model,
        sequence,
        feature_baseline,
    )

    print("=" * 65)
    print("LSTM FEATURE OCCLUSION ANALYSIS")
    print("=" * 65)
    print(f"Validation sequence: {sequence_index + 1}")
    print(f"Actual RUL: {actual_rul:.2f} cycles")
    print(f"Original predicted RUL: {original_prediction:.2f} cycles")
    print("Baseline: per-feature median across validation sequences")
    print("Cycle feature: preserved, not occluded")
    print()
    print("Top 10 features by absolute prediction change:")
    print(
        explanation[
            [
                "feature",
                "occluded_prediction",
                "signed_change_cycles",
                "importance_cycles",
            ]
        ]
        .head(10)
        .to_string(index=False, float_format="%.3f")
    )

    output_path = ARTIFACTS_DIR / "feature_occlusion_results.csv"
    explanation.to_csv(output_path, index=False)

    print()
    print(f"Full results saved to: {output_path}")
    print(
        "Interpretation: larger values indicate greater sensitivity "
        "to replacing a feature with its validation median."
    )
    print(
        "This is a model sensitivity analysis, not proof of causation "
        "or real-world sensor importance."
    )


if __name__ == "__main__":
    main()
