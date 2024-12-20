import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


def load_time_series_with_describe_features(
    dirname: Union[str, Path],
    chunk_size: int = 10,
) -> pd.DataFrame:
    def _process_file(file_path: str):
        df = pd.read_parquet(file_path)
        df.drop("step", axis=1, inplace=True)
        return (
            df.describe().values.reshape(-1),
            os.path.basename(os.path.dirname(file_path)).split("=")[1],
        )

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
            results = list(
                executor.map(_process_file, chunk),
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
