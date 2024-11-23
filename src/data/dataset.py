import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


def load_time_series_with_describe_features(dirname) -> pd.DataFrame:
    def _process_file(filename, dirname):
        df = pd.read_parquet(os.path.join(dirname, filename, "part-0.parquet"))
        df.drop("step", axis=1, inplace=True)
        return df.describe().values.reshape(-1), filename.split("=")[1]

    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(lambda fname: _process_file(fname, dirname), ids),
                total=len(ids),
            )
        )

    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    df["id"] = indexes

    return df
