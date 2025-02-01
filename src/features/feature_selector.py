import copy

import pandas as pd

from src import utils
from src.features import correlation
from src.features.univariate import univariate_feature_selection
from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


def select_features(train_df, test_df, save_path, correlation_threshold):
    selected_features = univariate_feature_selection(
        df=train_df.fillna(train_df.mean()).drop(
            columns=[utils.constants.TARGET_COLUMN_NAME]
        ),
        target=train_df[utils.constants.TARGET_COLUMN_NAME],
        save_path=save_path / "univariate_selected_features.txt",
    )
    train_df = subset_of_features(train_df, selected_features)

    selected_features = correlation.filter_features_by_threshold(
        df=train_df,
        threshold=correlation_threshold,
        save_path=save_path / "correlation_selected_features.txt",
    )
    train_df = subset_of_features(train_df, selected_features)
    test_df = subset_of_features(test_df, selected_features)

    return train_df, test_df


def subset_of_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    features = copy.deepcopy(features)
    LOGGER.info(f"Subsetting features: {len(features)} from {len(df.columns)}")
    if utils.constants.TARGET_COLUMN_NAME in df.columns:
        features.append(utils.constants.TARGET_COLUMN_NAME)
    return df[features]
