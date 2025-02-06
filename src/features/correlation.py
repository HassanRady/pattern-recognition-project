from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.data_manager import save_list_to_file, read_list_from_file, save_csv
from src.features.commons import filter_features
from src.logger import get_console_logger
from src.utils.others import skip_if_exists
from src import utils

LOGGER = get_console_logger(logger_name=__name__)


def get_feature_importance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Calculates feature importance based on correlation with the target variable.

    This function computes the absolute correlations of all features with the target variable,
    ranks them by importance, and returns a sorted DataFrame of feature importances.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing features and the target variable.
        target (str): The name of the target variable column.

    Returns:
        pd.DataFrame: A DataFrame with two columns - the feature names (index) and their
                      correlation-based importance, sorted in ascending order of importance.

    Notes:
        - The importance is calculated based on absolute correlation values.
        - The target variable is excluded from the output.
    """
    correlation_matrix = df.corr().abs()
    importance = correlation_matrix[target]
    feature_importance = importance.to_frame(name="importance")
    feature_importance.sort_values(by="importance", ascending=True, inplace=True)
    feature_importance.drop(index=target, inplace=True)
    return feature_importance


@skip_if_exists(reader=read_list_from_file)
def filter_features_by_threshold(
    df: pd.DataFrame, threshold: float, save_path: Optional[Path] = None
) -> list[str]:
    """
    Filters features based on a correlation importance threshold.

    This function calculates feature importance, filters features that meet or exceed
    the specified threshold, and optionally saves the selected features to a file.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing features and the target column.
        threshold (float): The minimum importance score for a feature to be selected.
        save_path (Optional[Path]): Path to save the list of selected features. If `None`,
                                    the list is not saved.

    Returns:
        list[str]: A list of selected feature names that meet or exceed the importance threshold.

    Notes:
        - Uses `get_feature_importance` to calculate the correlation-based importance of features.
        - Uses `filter_features` to select features based on the threshold.
        - The `@skip_if_exists` decorator skips execution if the saved file already exists
          with matching parameters.
    """
    LOGGER.info("Correlation feature selection started")
    feature_importance = get_feature_importance(df, utils.constants.SII_COLUMN_NAME)
    features = filter_features(feature_importance, threshold)
    LOGGER.info(
        f"Correlation feature selection: {len(features)} from {len(df.columns) - 1}"
    )
    if save_path:
        save_list_to_file(features, save_path)
        save_csv(
            feature_importance, save_path.parent / "correlation_feature_importance.csv"
        )
    return features
