import numpy as np

from src.models.data_ingestion import (
    ENGINE_COL,
    CYCLE_COL,
    TARGET_COL,
    FEATURE_COLUMNS,
)


WINDOW_SIZE = 50


def generate_sequences(df, window_size=WINDOW_SIZE):
    """
    Generate fixed-length sequences separately for each engine.

    Each input sequence contains `window_size` consecutive cycles.
    Its target is the RUL at the final cycle of that sequence.

    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed and scaled engine data.
    window_size : int
        Number of consecutive cycles in each sequence.

    Returns
    -------
    X : numpy.ndarray
        Shape (number_of_sequences, window_size, number_of_features).
    y : numpy.ndarray
        Shape (number_of_sequences,).
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero.")

    X_sequences = []
    y_sequences = []

    # Process each engine independently to prevent data leakage.
    for _, engine_df in df.groupby(ENGINE_COL, sort=True):

        engine_df = engine_df.sort_values(CYCLE_COL)

        features = engine_df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        targets = engine_df[TARGET_COL].to_numpy(
            dtype=np.float32
        )

        # Engines shorter than the window cannot produce sequences.
        if len(engine_df) < window_size:
            continue

        for end_idx in range(window_size, len(engine_df) + 1):

            start_idx = end_idx - window_size

            X_sequences.append(
                features[start_idx:end_idx]
            )

            # Predict the RUL at the final cycle in the window.
            y_sequences.append(
                targets[end_idx - 1]
            )

    if not X_sequences:
        raise ValueError(
            f"No sequences generated. "
            f"Check whether engines contain at least "
            f"{window_size} cycles."
        )

    X = np.stack(X_sequences).astype(np.float32)
    y = np.asarray(y_sequences, dtype=np.float32)

    return X, y


if __name__ == "__main__":
    from src.models.data_ingestion import (
        load_and_preprocess_cmapss,
        engine_wise_split,
        scale_features,
    )

    print("=" * 55)
    print("QUANTUMFLIX: SEQUENCE GENERATION")
    print("=" * 55)

    # Load and preprocess the raw dataset.
    df = load_and_preprocess_cmapss(
        "data/train_FD001.txt"
    )

    # Split by engine before fitting the scaler.
    train_df, val_df = engine_wise_split(df)

    # Fit the scaler on training engines only.
    train_df, val_df, scaler = scale_features(
        train_df,
        val_df,
    )

    # Generate sequences separately for each split.
    X_train, y_train = generate_sequences(train_df)
    X_val, y_val = generate_sequences(val_df)

    print("\nSEQUENCE GENERATION RESULTS")
    print("-" * 55)

    print("Window size:", WINDOW_SIZE)
    print("Number of features:", len(FEATURE_COLUMNS))

    print("\nTraining arrays:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\nValidation arrays:")
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    print("\nVERIFICATION")
    print("-" * 55)

    print(
        "Training shape is correct:",
        X_train.ndim == 3
        and X_train.shape[1:] == (
            WINDOW_SIZE,
            len(FEATURE_COLUMNS),
        ),
    )

    print(
        "Validation shape is correct:",
        X_val.ndim == 3
        and X_val.shape[1:] == (
            WINDOW_SIZE,
            len(FEATURE_COLUMNS),
        ),
    )

    print(
        "Training targets match sequences:",
        len(X_train) == len(y_train),
    )

    print(
        "Validation targets match sequences:",
        len(X_val) == len(y_val),
    )

    print("\nSequence generation completed successfully!")