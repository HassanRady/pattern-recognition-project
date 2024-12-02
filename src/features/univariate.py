from pathlib import Path
from typing import Optional

import pandas as pd
from tsfresh import select_features

from src.data.data_manager import save_list_to_file, read_list_from_file
from src.logger import get_console_logger
from src.utils.others import skip_if_exists

LOGGER = get_console_logger(logger_name=__name__)


@skip_if_exists(reader=read_list_from_file)
def univariate_feature_selection(
    df: pd.DataFrame, target: pd.Series, save_path: Optional[Path] = None
) -> list[str]:
    """
    Performs univariate feature selection for regression tasks.

    This function selects features from the input DataFrame that are most relevant
    to the target variable using univariate feature selection from tsfresh. The selected feature
    names are optionally saved to a file.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing feature columns.
        target (pd.Series): The target variable for the regression task.
        save_path (Optional[Path]): Path to save the list of selected feature names.
                                    If `None`, the list is not saved.

    Returns:
        list[str]: A list of selected feature names.

    Notes:
        - Uses the `select_features` function to perform feature selection.
        - Logs the number of selected features and the total number of features.
        - The `@skip_if_exists` decorator skips execution if the file already exists
          and matches the parameters.
    """
    LOGGER.info("Univariate feature selection started")
    features = select_features(df, target).columns.to_list()
    LOGGER.info(
        f"Univariate feature selection: {len(features)} from {len(df.columns) - 1}"
    )
    if save_path:
        save_list_to_file(features, save_path)
    return features
