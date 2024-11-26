import threading
from pathlib import Path
from typing import Any, Tuple, Union, List, Optional

import joblib
import pandas as pd
import numpy.typing as npt

from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


def read_parquet(
    path: Union[Path, str],
    verbose: Optional[bool] = True,
) -> pd.DataFrame:
    if verbose:
        LOGGER.info(f"Reading parquet from {path}")
    return pd.read_parquet(path)


def read_csv(
    path: Union[Path, str],
    index_cols: Optional[Union[int, List[int]]] = 0,
    verbose: Optional[bool] = True,
) -> pd.DataFrame:
    if verbose:
        LOGGER.info(f"Reading csv from {path}")
    return pd.read_csv(path, index_col=index_cols)


def save_csv(df: pd.DataFrame, path: Union[Path, str]):
    LOGGER.info(f"Saving csv to {path}")
    path = Path(path) if isinstance(path, str) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def save_model(model: Any, path: Path) -> None:
    LOGGER.info(f"Saving model to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.suffix == ".joblib":
        path = path.with_suffix(".joblib")
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    LOGGER.info(f"Loading model from {path}")
    if not path.suffix == ".joblib":
        path = path.with_suffix(".joblib")
    return joblib.load(path)


def prepare_data_for_estimator(
    df: pd.DataFrame, target: str
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    return df.drop(columns=[target]).values, df[target].values


def save_scores(scores_dfs: dict[str, pd.DataFrame], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, df in scores_dfs.items():
        LOGGER.info(f"Saving {name} scores to {path / f'{name}.csv'}")
        if not (path / f"{name}.csv").exists():
            df.to_csv(path / f"{name}.csv")
        else:
            df.to_csv(path / f"{name}.csv", mode="a", header=False)
