import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from src.models.data_ingestion import (
    load_and_preprocess_cmapss,
    engine_wise_split,
    scale_features,
)

from src.features.build_sequences import generate_sequences


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/train_FD001.txt"
OUTPUT_DIR = "artifacts"
RANDOM_STATE = 42


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_train, y_train, X_val, y_val, name):
    """
    Train a model and evaluate its predictions using MAE and RMSE.
    Classical ML models use the final timestep of each sequence.
    """

    print(f"\nTraining {name}...")

    # Convert 3D sequences into 2D feature vectors.
    # Each row represents the final cycle of a 50-cycle window.
    X_train_2d = X_train[:, -1, :]
    X_val_2d = X_val[:, -1, :]

    model.fit(X_train_2d, y_train)

    predictions = model.predict(X_val_2d)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))

    print(f"{name} results:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    return model, {
        "model": name,
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("QUANTUMFLIX: CLASSICAL ML BASELINES")
    print("=" * 60)

    # 1. Load and preprocess data.
    print("\nLoading dataset...")
    df = load_and_preprocess_cmapss(DATA_PATH)

    # 2. Split by engine.
    print("Splitting data by engine...")
    train_df, val_df = engine_wise_split(df)

    # 3. Fit scaler on training data only.
    print("Scaling features...")
    train_df, val_df, scaler = scale_features(
        train_df,
        val_df,
    )

    # 4. Generate 50-cycle sequences.
    print("Generating sequences...")
    X_train, y_train = generate_sequences(train_df)
    X_val, y_val = generate_sequences(val_df)

    print("\nSequence shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:  ", X_val.shape)
    print("y_val:  ", y_val.shape)

    # 5. Save arrays for the LSTM teammate.
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "sequence_data.npz"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    joblib.dump(
        scaler,
        os.path.join(OUTPUT_DIR, "feature_scaler.joblib"),
    )

    print("\nSaved sequence arrays and scaler.")

    # 6. Define classical models.
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
        ),
    }

    # 7. Train and evaluate.
    results = []

    for name, model in models.items():

        trained_model, metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            name,
        )

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            trained_model,
            os.path.join(OUTPUT_DIR, filename),
        )

        results.append(metrics)

    # 8. Save metrics.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    results_path = os.path.join(
        OUTPUT_DIR,
        "baseline_metrics.csv",
    )

    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL BASELINE COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]["model"]
    print(f"\nBest model by RMSE: {best_model}")

    print("\nArtifacts saved in:", OUTPUT_DIR)
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()