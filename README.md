# QUANTUMFLIX: Predictive Maintenance Using Machine Learning

## Overview

QUANTUMFLIX is a machine learning project focused on predictive maintenance
and Remaining Useful Life (RUL) estimation using NASA's C-MAPSS FD001 dataset.

The project explores classical machine learning baselines and a sequence-based
deep learning approach to estimate how many operational cycles remain before
engine failure.

## Dataset

The project uses the NASA C-MAPSS FD001 dataset.

The raw training file is expected at:

`data/train_FD001.txt`

The dataset contains engine identifiers, operational cycles, three operational
settings, and 21 sensor measurements.

Raw datasets are excluded from version control. Place the dataset in the
expected location before running the pipeline.

## Role 1: Data Processing and Classical ML

Role 1 implements:

- Dataset ingestion and preprocessing.
- Removal of seven selected sensors.
- Uncapped Remaining Useful Life (RUL) calculation.
- Engine-wise 80/20 training and validation split.
- Feature scaling fitted on training data only.
- Generation of 50-cycle sequences.
- Linear Regression, Random Forest, and XGBoost baselines.
- Evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Retained Features

The 18 features are:

1. cycle
2. op_setting_1
3. op_setting_2
4. op_setting_3
5. sensor_2
6. sensor_3
7. sensor_4
8. sensor_7
9. sensor_8
10. sensor_9
11. sensor_11
12. sensor_12
13. sensor_13
14. sensor_14
15. sensor_15
16. sensor_17
17. sensor_20
18. sensor_21

The cycle feature is left unscaled. Operational settings and retained sensor
features are scaled using MinMaxScaler.

### Sequence Generation

Each sequence contains 50 consecutive cycles from a single engine.

The target is the RUL at the final cycle of the sequence. Sequences never cross
engine boundaries.

Expected array formats:

| Array | Shape |
|---|---|
| X_train | (N, 50, 18) |
| y_train | (N,) |
| X_val | (M, 50, 18) |
| y_val | (M,) |

## Baseline Results

The following results were obtained using an engine-wise validation split of
the training dataset.

| Model | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 21.3850 | 27.3663 |
| Random Forest | 20.5529 | 28.0397 |
| XGBoost | 20.2959 | 27.6585 |

Lower MAE and RMSE indicate better predictive performance.

XGBoost achieved the lowest MAE, while Linear Regression achieved the lowest
RMSE in this experiment.

These results are validation results, not official test-set results.

## Project Structure

```text
RUL-Predictive-Maintenance/
├── data/
│   └── train_FD001.txt
├── src/
│   ├── features/
│   │   └── build_sequences.py
│   └── models/
│       ├── data_ingestion.py
│       └── baseline_train.py
├── artifacts/
├── requirements.txt
├── .gitignore
└── README.md