import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. FEATURE CONFIGURATION
# ============================================================

ENGINE_COL = "engine_id"
CYCLE_COL = "cycle"
TARGET_COL = "RUL"

OP_SETTING_COLS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
]

# Sensors specified for removal in the QUANTUMFLIX data contract
DROP_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# Retain the remaining 14 sensors in ascending sensor-number order
RETAINED_SENSORS = [
    f"sensor_{i}"
    for i in range(1, 22)
    if f"sensor_{i}" not in DROP_SENSORS
]

# Exact feature order expected by Role 2
FEATURE_COLUMNS = [
    CYCLE_COL,
    *OP_SETTING_COLS,
    *RETAINED_SENSORS,
]

# Scale the three operational settings and 14 retained sensors.
# Keep cycle in its original units.
SCALED_COLUMNS = OP_SETTING_COLS + RETAINED_SENSORS


# ============================================================
# 2. LOAD AND PREPROCESS THE DATASET
# ============================================================

def load_and_preprocess_cmapss(file_path):

    # Define the original 26 columns:
    # 1 engine ID + 1 cycle + 3 operational settings + 21 sensors
    columns = [
        ENGINE_COL,
        CYCLE_COL,
        *OP_SETTING_COLS,
        *[f"sensor_{i}" for i in range(1, 22)],
    ]

    # Load the headerless, whitespace-separated dataset
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    # Validate the dataset
    if df.empty:
        raise ValueError("The dataset is empty.")

    if df.isnull().any().any():
        raise ValueError(
            "The dataset contains missing values. "
            "Check the raw C-MAPSS file."
        )

    # Remove the seven specified sensors
    df = df.drop(columns=DROP_SENSORS)

    # Calculate uncapped RUL:
    # RUL = Maximum cycle of engine - Current cycle
    max_cycles = df.groupby(ENGINE_COL)[CYCLE_COL].transform("max")

    df[TARGET_COL] = max_cycles - df[CYCLE_COL]

    # Sort chronologically within each engine
    df = df.sort_values(
        [ENGINE_COL, CYCLE_COL]
    ).reset_index(drop=True)

    return df


# ============================================================
# 3. SPLIT DATA BY ENGINE ID
# ============================================================

def engine_wise_split(df, test_size=0.2):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42,
    )

    # All observations belonging to one engine stay together
    train_indices, val_indices = next(
        splitter.split(
            df,
            groups=df[ENGINE_COL],
        )
    )

    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()

    # Sort each split by engine and operating cycle
    train_df = train_df.sort_values(
        [ENGINE_COL, CYCLE_COL]
    ).reset_index(drop=True)

    val_df = val_df.sort_values(
        [ENGINE_COL, CYCLE_COL]
    ).reset_index(drop=True)

    # Verify that no engine appears in both splits
    train_engines = set(train_df[ENGINE_COL].unique())
    val_engines = set(val_df[ENGINE_COL].unique())

    if not train_engines.isdisjoint(val_engines):
        raise ValueError(
            "Data leakage detected: an engine appears "
            "in both training and validation data."
        )

    return train_df, val_df


# ============================================================
# 4. SCALE FEATURES WITHOUT DATA LEAKAGE
# ============================================================

def scale_features(train_df, val_df):

    # Work on copies to avoid modifying the original DataFrames
    train_df = train_df.copy()
    val_df = val_df.copy()

    # Create the scaler
    scaler = MinMaxScaler()

    # IMPORTANT:
    # Convert integer columns to floating-point BEFORE scaling.
    # This prevents pandas from rejecting decimal values.
    train_df[SCALED_COLUMNS] = train_df[
        SCALED_COLUMNS
    ].astype(float)

    val_df[SCALED_COLUMNS] = val_df[
        SCALED_COLUMNS
    ].astype(float)

    # Fit ONLY on training data
    train_df.loc[:, SCALED_COLUMNS] = scaler.fit_transform(
        train_df[SCALED_COLUMNS]
    )

    # Transform validation data using the training-fitted scaler
    val_df.loc[:, SCALED_COLUMNS] = scaler.transform(
        val_df[SCALED_COLUMNS]
    )

    return train_df, val_df, scaler


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("QUANTUMFLIX: DATA INGESTION AND PREPROCESSING")
    print("=" * 55)

    # Load the raw training dataset
    df = load_and_preprocess_cmapss(
        "data/train_FD001.txt"
    )

    print("\nDataset loaded successfully.")

    # Split by engine ID
    train_df, val_df = engine_wise_split(df)

    # Scale the features
    train_df, val_df, scaler = scale_features(
        train_df,
        val_df,
    )

    # Verify engine separation
    train_engines = set(train_df[ENGINE_COL].unique())
    val_engines = set(val_df[ENGINE_COL].unique())

    assert train_engines.isdisjoint(val_engines)

    # Display results
    print("\nPREPROCESSING RESULTS")
    print("-" * 55)

    print(
        f"Total engines: "
        f"{df[ENGINE_COL].nunique()}"
    )

    print(
        f"Training engines: "
        f"{train_df[ENGINE_COL].nunique()}"
    )

    print(
        f"Validation engines: "
        f"{val_df[ENGINE_COL].nunique()}"
    )

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Validation rows: {len(val_df)}"
    )

    print(
        f"Input feature count: {len(FEATURE_COLUMNS)}"
    )

    print("\nFeature order:")
    for index, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"{index:2}. {feature}")

    print("\nRUL ranges:")
    print(
        f"Training RUL: "
        f"{train_df[TARGET_COL].min()} to "
        f"{train_df[TARGET_COL].max()}"
    )

    print(
        f"Validation RUL: "
        f"{val_df[TARGET_COL].min()} to "
        f"{val_df[TARGET_COL].max()}"
    )

    print("\nVERIFICATION")
    print("-" * 55)

    print("Feature count is 18:", len(FEATURE_COLUMNS) == 18)

    print(
        "No engine leakage:",
        train_engines.isdisjoint(val_engines),
    )

    print(
        "Scaler fitted on training engines only: True"
    )

    print("\nData ingestion and preprocessing completed!")