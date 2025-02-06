import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


def time_features(df):
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    features = [
        df["non-wear_flag"].mean(),
        df["enmo"][df["enmo"] >= 0.05].sum(),
    ]

    night = (df["hours"] >= 22) | (df["hours"] <= 5)
    day = (df["hours"] <= 20) & (df["hours"] >= 7)
    no_mask = np.ones(len(df), dtype=bool)

    keys = ["enmo", "anglez", "light", "battery_voltage"]
    masks = [no_mask, night, day]

    def extract_stats(data):
        return [
            data.mean(),
            data.std(),
            data.max(),
            data.min(),
            data.diff().mean(),
            data.diff().std(),
        ]

    for key in keys:
        for mask in masks:
            filtered_data = df.loc[mask, key]
            features.extend(extract_stats(filtered_data))

    return features


def load_time_series(
    dirname: Union[str, Path],
    chunk_size: int = 10,
    stats: bool = False,
) -> pd.DataFrame:
    def _process_file(file_path: str):
        df = pd.read_parquet(file_path)
        df.drop("step", axis=1, inplace=True)
        return (
            df.describe().values.reshape(-1),
            os.path.basename(os.path.dirname(file_path)).split("=")[1],
        )

    def _process_file2(file_path: str):
        df = pd.read_parquet(file_path)
        df.drop("step", axis=1, inplace=True)
        return time_features(df), os.path.basename(os.path.dirname(file_path)).split(
            "="
        )[1]

    file_paths = [
        os.path.join(entry.path, "part-0.parquet")
        for entry in os.scandir(dirname)
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, "part-0.parquet"))
    ]

    all_results = []
    for i in range(0, len(file_paths), chunk_size):
        LOGGER.info(
            f"Processing chunk {i // chunk_size + 1}/{(len(file_paths) + chunk_size - 1) // chunk_size}"
        )
        chunk = file_paths[i : i + chunk_size]

        with ThreadPoolExecutor() as executor:
            if stats:
                results = list(
                    executor.map(_process_file, chunk),
                )
            else:
                results = list(
                    executor.map(_process_file2, chunk),
                )

        all_results.extend(results)

    stats, indexes = zip(*all_results)
    stats_df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    stats_df["id"] = indexes

    return stats_df


def read_tabular_dataset(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info(f"Reading tabular dataset from {path}")
    train_path = path / "train.csv"
    train_df = pd.read_csv(train_path)
    train_df.set_index("id", inplace=True)
    test_path = path / "test.csv"
    test_df = pd.read_csv(test_path)
    test_df.set_index("id", inplace=True)
    return train_df, test_df
