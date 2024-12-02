import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from data.data_manager import read_csv
from features.commons import filter_features
from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


def get_feature_importance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Calculate feature importance using Mutual Information Regression.

    This function computes the importance of each feature by measuring the mutual
    information between the features and the target variable. The importance scores
    are then normalized to be between 0 and 1, and the features are sorted by their
    importance in ascending order.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the features and the target.
        target (str): The name of the target column in `df`.

    Returns:
        pd.DataFrame: A DataFrame with the features and their corresponding importance scores,
                      sorted in ascending order.

    Notes:
        - The importance scores are normalized so that the smallest value is 0 and the largest is 1.
    """
    importance = mutual_info_regression(df.drop(columns=[target]), df[target])
    feature_importance = pd.DataFrame(
        importance, index=df.drop(columns=[target]).columns, columns=["importance"]
    )

    # Normalize feature importance so that it is between 0 and 1 to filter features
    feature_importance = (feature_importance - feature_importance.min()) / (
        feature_importance.max() - feature_importance.min()
    )
    feature_importance.sort_values(by="importance", ascending=True, inplace=True)
    return feature_importance


def filter_features_by_threshold(
    df: pd.DataFrame, target: str, threshold: float
) -> list[str]:
    """
    Filter features by Mutual Information Regression importance threshold.

    This function calculates feature importance using Mutual Information Regression
    and then filters the features that have an importance score greater than or equal
    to the specified threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the features and the target.
        target (str): The name of the target column in `df`.
        threshold (float): The minimum importance score required for a feature to be selected.

    Returns:
        list[str]: A list of feature names that meet or exceed the importance threshold.

    Notes:
        - Relies on `get_feature_importance` to compute Mutual Information Regression based importance.
        - Only features with Mutual Information Regression values above the threshold are selected.
    """
    LOGGER.info("Mutual Information Regression feature selection started")
    feature_importance = get_feature_importance(df, target)
    features = filter_features(feature_importance, threshold)
    LOGGER.info(
        f"Mutual Information Regression feature selection: {len(features)} from {len(df.columns) - 1}"
    )
    return features

