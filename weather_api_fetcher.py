from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = DATA_DIR / "api_weather"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Coordinates used for the two plants. If official metadata is available later,
# update only this mapping.
PLANT_COORDINATES = {
    1: {"latitude": 14.815, "longitude": 78.287},
    2: {"latitude": 19.997, "longitude": 73.790},
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "sunshine_duration",
]


def resolve_dataset_path(filename: str) -> Path:
    dataset_path = DATA_DIR / filename
    if dataset_path.exists():
        return dataset_path

    fallback_path = BASE_DIR / filename
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"Khong tim thay dataset: {filename}")


def parse_generation_datetime(series: pd.Series, plant_no: int) -> pd.Series:
    if plant_no == 1:
        return pd.to_datetime(series, format="%d-%m-%Y %H:%M", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def get_dataset_date_range(plant_no: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    generation = pd.read_csv(resolve_dataset_path(f"Plant_{plant_no}_Generation_Data.csv"))
    weather = pd.read_csv(resolve_dataset_path(f"Plant_{plant_no}_Weather_Sensor_Data.csv"))

    generation["DATE_TIME"] = parse_generation_datetime(generation["DATE_TIME"], plant_no)
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")

    all_times = pd.concat([generation["DATE_TIME"], weather["DATE_TIME"]], ignore_index=True)
    if all_times.isna().any():
        raise ValueError(f"Loi parse DATE_TIME o Plant {plant_no}")

    return all_times.min(), all_times.max()


def fetch_open_meteo_hourly(plant_no: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    coordinates = PLANT_COORDINATES[plant_no]
    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo API error"))
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError("Open-Meteo response khong co hourly time")

    hourly = pd.DataFrame({"DATE_TIME": pd.to_datetime(payload["hourly"]["time"])})
    for variable in HOURLY_VARIABLES:
        if variable not in payload["hourly"]:
            raise RuntimeError(f"Open-Meteo response thieu bien: {variable}")
        hourly[f"OM_{variable.upper()}"] = payload["hourly"][variable]

    hourly["PLANT_NO"] = plant_no
    hourly["LATITUDE"] = coordinates["latitude"]
    hourly["LONGITUDE"] = coordinates["longitude"]
    hourly["OPEN_METEO_TIMEZONE"] = payload.get("timezone")
    hourly["OPEN_METEO_UTC_OFFSET_SECONDS"] = payload.get("utc_offset_seconds")
    return hourly


def align_to_dataset_frequency(hourly: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    full_index = pd.date_range(start=start, end=end, freq="15min")
    metadata_cols = [
        "PLANT_NO",
        "LATITUDE",
        "LONGITUDE",
        "OPEN_METEO_TIMEZONE",
        "OPEN_METEO_UTC_OFFSET_SECONDS",
    ]
    value_cols = [col for col in hourly.columns if col not in metadata_cols + ["DATE_TIME"]]

    aligned = hourly.set_index("DATE_TIME")[value_cols].reindex(full_index, method="ffill")
    aligned = aligned.reset_index().rename(columns={"index": "DATE_TIME"})

    for col in metadata_cols:
        aligned[col] = hourly[col].iloc[0]

    ordered_cols = ["PLANT_NO", "DATE_TIME", *value_cols, "LATITUDE", "LONGITUDE"]
    return aligned[ordered_cols + ["OPEN_METEO_TIMEZONE", "OPEN_METEO_UTC_OFFSET_SECONDS"]]


def save_weather_csv(plant_no: int, output_dir: Path = OUTPUT_DIR) -> Path:
    if plant_no not in PLANT_COORDINATES:
        raise ValueError(f"Plant khong hop le: {plant_no}")

    output_dir.mkdir(parents=True, exist_ok=True)
    start, end = get_dataset_date_range(plant_no)
    hourly = fetch_open_meteo_hourly(plant_no, start, end)
    aligned = align_to_dataset_frequency(hourly, start, end)

    output_path = output_dir / f"open_meteo_weather_plant_{plant_no}.csv"
    aligned.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lay du lieu Open-Meteo theo plant va luu thanh CSV da can moc 15 phut."
    )
    parser.add_argument(
        "--plant",
        type=int,
        choices=[1, 2],
        action="append",
        help="Plant can lay du lieu. Co the truyen nhieu lan. Mac dinh lay ca 1 va 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Thu muc luu CSV Open-Meteo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plants = args.plant or sorted(PLANT_COORDINATES)

    for plant_no in plants:
        output_path = save_weather_csv(plant_no, args.output_dir)
        print(f"Da luu Plant {plant_no}: {output_path}")


if __name__ == "__main__":
    main()
