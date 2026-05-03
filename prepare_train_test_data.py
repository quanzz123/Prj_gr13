from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_INPUT = OUTPUT_DIR / "eda_all_plants_merged.csv"

TARGET_COLUMN = "AC_POWER_TOTAL"

BASE_FEATURES = [
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "HOUR",
    "MINUTE",
    "DAYOFWEEK",
    "DAYOFYEAR",
    "MONTH",
    "HOUR_SIN",
    "HOUR_COS",
    "IS_DAYLIGHT",
]

IDENTIFIER_COLUMNS = ["PLANT_ID", "DATE_TIME", "PLANT_NO"]


def load_merged_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay {input_path}. Hay chay notebook EDASolarPowerGenerationData truoc."
        )

    df = pd.read_csv(input_path, parse_dates=["DATE_TIME"])
    required = ["DATE_TIME", "PLANT_NO", TARGET_COLUMN, *BASE_FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Thieu cot bat buoc trong input: {missing}")

    return df


def select_feature_columns(df: pd.DataFrame, use_open_meteo: bool) -> list[str]:
    features = [col for col in BASE_FEATURES if col in df.columns]

    if use_open_meteo:
        open_meteo_features = [
            col
            for col in df.columns
            if col.startswith("OM_") and pd.api.types.is_numeric_dtype(df[col])
        ]
        features.extend(open_meteo_features)

    return features


def time_train_test_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size phai nam trong khoang (0, 1)")

    sorted_df = df.sort_values(["DATE_TIME", "PLANT_NO"]).reset_index(drop=True)
    split_index = int(len(sorted_df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(sorted_df):
        raise ValueError("Split khong hop le, hay chon test_size khac")

    return sorted_df.iloc[:split_index].copy(), sorted_df.iloc[split_index:].copy()


def fit_median_imputer(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    medians = train_df[feature_columns].median(numeric_only=True)
    return {col: float(medians[col]) for col in feature_columns}


def apply_median_imputer(df: pd.DataFrame, feature_columns: list[str], medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        out[col] = out[col].fillna(medians[col])
    return out


def fit_standard_scaler(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    scaler: dict[str, dict[str, float]] = {}
    for col in feature_columns:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std(ddof=0))
        if std == 0:
            std = 1.0
        scaler[col] = {"mean": mean, "std": std}
    return scaler


def apply_standard_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        out[col] = (out[col] - scaler[col]["mean"]) / scaler[col]["std"]
    return out


def add_plant_one_hot(df: pd.DataFrame, plant_values: list[int]) -> pd.DataFrame:
    out = df.copy()
    for plant_no in plant_values:
        out[f"PLANT_NO_{plant_no}"] = (out["PLANT_NO"] == plant_no).astype(int)
    return out


def prepare_train_test(
    input_path: Path,
    output_dir: Path,
    test_size: float,
    use_open_meteo: bool,
) -> dict:
    df = load_merged_data(input_path)
    feature_columns = select_feature_columns(df, use_open_meteo)

    model_df = df.dropna(subset=[TARGET_COLUMN]).copy()
    train_raw, test_raw = time_train_test_split(model_df, test_size)

    medians = fit_median_imputer(train_raw, feature_columns)
    train_imputed = apply_median_imputer(train_raw, feature_columns, medians)
    test_imputed = apply_median_imputer(test_raw, feature_columns, medians)

    scaler = fit_standard_scaler(train_imputed, feature_columns)
    train_scaled = apply_standard_scaler(train_imputed, feature_columns, scaler)
    test_scaled = apply_standard_scaler(test_imputed, feature_columns, scaler)

    plant_values = sorted(int(value) for value in model_df["PLANT_NO"].dropna().unique())
    train_ready = add_plant_one_hot(train_scaled, plant_values)
    test_ready = add_plant_one_hot(test_scaled, plant_values)

    plant_feature_columns = [f"PLANT_NO_{plant_no}" for plant_no in plant_values]
    final_columns = [
        *IDENTIFIER_COLUMNS,
        *feature_columns,
        *plant_feature_columns,
        TARGET_COLUMN,
    ]
    final_columns = [col for col in final_columns if col in train_ready.columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_data_normalized.csv"
    test_path = output_dir / "test_data_normalized.csv"
    metadata_path = output_dir / "train_test_metadata.json"

    train_ready[final_columns].to_csv(train_path, index=False)
    test_ready[final_columns].to_csv(test_path, index=False)

    metadata = {
        "input_path": str(input_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "target_column": TARGET_COLUMN,
        "split_method": "time_ordered",
        "test_size": test_size,
        "train_rows": int(len(train_ready)),
        "test_rows": int(len(test_ready)),
        "train_date_min": str(train_ready["DATE_TIME"].min()),
        "train_date_max": str(train_ready["DATE_TIME"].max()),
        "test_date_min": str(test_ready["DATE_TIME"].min()),
        "test_date_max": str(test_ready["DATE_TIME"].max()),
        "numeric_feature_columns_scaled": feature_columns,
        "plant_one_hot_columns": plant_feature_columns,
        "dropped_default_leakage_columns": [
            "DC_POWER_TOTAL",
            "DAILY_YIELD_TOTAL",
            "TOTAL_YIELD_TOTAL",
            "ZERO_AC_COUNT",
            "ZERO_AC_RATIO",
            "ACTIVE_SOURCE_COUNT",
        ],
        "imputer": {"strategy": "median", "values": medians},
        "scaler": {"type": "standard", "values": scaler},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chuan hoa feature va chia train/test theo thoi gian, khong train model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="File merged tu EDA notebook.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Thu muc luu train/test.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Ty le test o cuoi chuoi thoi gian.")
    parser.add_argument(
        "--no-open-meteo",
        action="store_true",
        help="Khong dung cac cot OM_* lam feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_train_test(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        use_open_meteo=not args.no_open_meteo,
    )
    print(f"Da luu train: {metadata['train_path']}")
    print(f"Da luu test: {metadata['test_path']}")
    print(f"Da luu metadata: {args.output_dir / 'train_test_metadata.json'}")
    print(f"Train rows: {metadata['train_rows']}; Test rows: {metadata['test_rows']}")


if __name__ == "__main__":
    main()
