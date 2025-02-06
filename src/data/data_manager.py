from pathlib import Path
from typing import Any, Tuple, Union, List, Optional

import joblib
import numpy as np
import pandas as pd

from src import utils
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
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        df.drop(columns=[utils.constants.SII_COLUMN_NAME]).values,
        df[utils.constants.SII_COLUMN_NAME].values,
        df[utils.constants.PCIAT_TOTAL_CULUMN_NAME].values,
    )


def save_scores(scores_dfs: dict[str, pd.DataFrame], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, df in scores_dfs.items():
        LOGGER.info(f"Saving {name} scores to {path / f'{name}.csv'}")
        if not (path / f"{name}.csv").exists():
            df.to_csv(path / f"{name}.csv")
        else:
            df.to_csv(path / f"{name}.csv", mode="a", header=False)


def stratified_fold_generator(
    df: pd.DataFrame, n_folds: int = 5
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    x, y = prepare_data_for_estimator(df)
    for n, train_index, test_index in enumerate(skf.split(x, y)):
        yield n, df.iloc[train_index], df.iloc[test_index]


def read_list_from_file(path: Path) -> List[str]:
    """
    Reads a list of strings from a file.

    This function reads the content of the specified file, evaluates it as a Python
    expression, and returns it as a list of strings.

    Parameters:
        path (Path): The file path to read the list from.

    Returns:
        List[str]: The list of strings read from the file.

    Notes:
        - Uses `eval` to parse the content of the file, so ensure the file contains
          a valid Python list representation.
    """
    LOGGER.info(f"Reading list from {path}")
    with path.open("r") as file:
        return eval(file.read())


def save_list_to_file(x: List[str], path: Path) -> None:
    """
    Saves a list of strings to a file.

    This function writes the list of strings to the specified file, ensuring the directory
    exists before saving.

    Parameters:
        x (List[str]): The list of strings to save.
        path (Path): The file path where the list will be saved.

    Notes:
        - Creates parent directories if they do not exist.
        - Writes the list as a string representation.
    """
    LOGGER.info(f"Saving list to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(x))
