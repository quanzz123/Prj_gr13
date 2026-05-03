from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
TRAIN_PATH = OUTPUT_DIR / "train_data_normalized.csv"
TEST_PATH = OUTPUT_DIR / "test_data_normalized.csv"
METADATA_PATH = OUTPUT_DIR / "train_test_metadata.json"
TARGET_COLUMN = "AC_POWER_TOTAL"
ID_COLUMNS = {"PLANT_ID", "DATE_TIME", "PLANT_NO"}


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Khong tim thay train/test da chuan hoa. Hay chay prepare_train_test_data.py "
            "hoac notebook EDASolarPowerGenerationData truoc."
        )

    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["DATE_TIME"])
    test_df = pd.read_csv(TEST_PATH, parse_dates=["DATE_TIME"])
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return train_df, test_df, metadata


def get_feature_columns(train_df: pd.DataFrame, metadata: dict) -> list[str]:
    metadata_features = metadata.get("numeric_feature_columns_scaled", [])
    metadata_plant_features = metadata.get("plant_one_hot_columns", [])
    feature_columns = [col for col in [*metadata_features, *metadata_plant_features] if col in train_df.columns]

    if feature_columns:
        return feature_columns

    return [
        col
        for col in train_df.columns
        if col not in ID_COLUMNS and col != TARGET_COLUMN and pd.api.types.is_numeric_dtype(train_df[col])
    ]


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(math.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_case_architectures(first_layer_neurons: int) -> list[dict]:
    second_third_layer_values = [(1, 1), (1, 3), (1, 6), (3, 1), (3, 3), (3, 6), (6, 1), (6, 3), (6, 6)]
    return [
        {
            "case": case_no,
            "hidden_layer_sizes": (first_layer_neurons, second_layer, third_layer),
        }
        for case_no, (second_layer, third_layer) in enumerate(second_third_layer_values, start=1)
    ]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    string_df = df.copy()
    for col in string_df.columns:
        if pd.api.types.is_float_dtype(string_df[col]):
            string_df[col] = string_df[col].map(lambda value: f"{value:.4f}")
        else:
            string_df[col] = string_df[col].astype(str)

    header = "| " + " | ".join(string_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(string_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in string_df.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def train_one_case(
    X_train: pd.DataFrame,
    y_train_scaled: np.ndarray,
    X_test: pd.DataFrame,
    y_scaler: StandardScaler,
    hidden_layer_sizes: tuple[int, int, int],
) -> tuple[np.ndarray, int, float]:
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=2000,
        tol=1e-5,
        n_iter_no_change=50,
        random_state=42,
        early_stopping=False,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train_scaled)

    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
    return y_pred, int(model.n_iter_), float(model.loss_)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df, metadata = load_train_test()
    feature_columns = get_feature_columns(train_df, metadata)

    input_count = len(feature_columns)
    output_count = 1
    first_layer_neurons = math.ceil((input_count + output_count) / 2)
    architectures = build_case_architectures(first_layer_neurons)

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()

    metric_rows = []
    prediction_df = test_df[["DATE_TIME", "PLANT_NO", TARGET_COLUMN]].copy()

    for case in architectures:
        y_pred, iterations, final_loss = train_one_case(
            X_train=X_train,
            y_train_scaled=y_train_scaled,
            X_test=X_test,
            y_scaler=y_scaler,
            hidden_layer_sizes=case["hidden_layer_sizes"],
        )
        prediction_df[f"CASE_{case['case']}_PREDICTED_AC_POWER"] = y_pred

        base_row = {
            "case": case["case"],
            "hidden_layer_1": case["hidden_layer_sizes"][0],
            "hidden_layer_2": case["hidden_layer_sizes"][1],
            "hidden_layer_3": case["hidden_layer_sizes"][2],
            "scope": "overall",
            "iterations": iterations,
            "final_loss_scaled_target": final_loss,
        }
        metric_rows.append({**base_row, **compute_metrics(y_test, y_pred)})

        for plant_no in sorted(test_df["PLANT_NO"].dropna().unique()):
            mask = test_df["PLANT_NO"] == plant_no
            plant_row = {**base_row, "scope": f"plant_{int(plant_no)}"}
            metric_rows.append({**plant_row, **compute_metrics(y_test[mask], y_pred[mask])})

    metrics_df = pd.DataFrame(metric_rows)
    overall_df = metrics_df[metrics_df["scope"] == "overall"].copy()
    overall_df["rank_by_rmse"] = overall_df["RMSE"].rank(method="dense").astype(int)
    metrics_df = metrics_df.merge(
        overall_df[["case", "rank_by_rmse"]],
        on="case",
        how="left",
    ).sort_values(["rank_by_rmse", "case", "scope"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_DIR / "mlp_case_metrics.csv", index=False)
    prediction_df.to_csv(OUTPUT_DIR / "mlp_case_predictions.csv", index=False)

    best = overall_df.sort_values("RMSE").iloc[0]
    report = [
        "# MLP Case Comparison",
        "",
        f"- Input features: {input_count}",
        f"- Output neurons: {output_count}",
        f"- a = ceil(({input_count} + {output_count}) / 2) = {first_layer_neurons}",
        f"- Best overall case by RMSE: Case {int(best['case'])}",
        "",
        "## Overall Metrics",
        "",
        dataframe_to_markdown(
            overall_df.sort_values("RMSE")[
                ["case", "hidden_layer_1", "hidden_layer_2", "hidden_layer_3", "MAE", "MSE", "RMSE", "R2"]
            ]
        ),
        "",
    ]
    (OUTPUT_DIR / "mlp_case_comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics_df, prediction_df


def main() -> None:
    metrics_df, _ = run_experiment()
    overall = metrics_df[metrics_df["scope"] == "overall"].sort_values("RMSE")
    print(overall[["case", "hidden_layer_1", "hidden_layer_2", "hidden_layer_3", "MAE", "MSE", "RMSE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
