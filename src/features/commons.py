import pandas as pd


def filter_features(feature_importance: pd.DataFrame, threshold: float) -> list[str]:
    """
    Filters features based on an importance threshold.

    This function selects features with an importance score greater than or equal
    to the specified threshold and returns their names.

    Parameters:
        feature_importance (pd.DataFrame): DataFrame containing feature names as the index
                                           and their importance scores in a column named "importance".
        threshold (float): The minimum importance score required for a feature to be selected.

    Returns:
        list[str]: A list of feature names that meet or exceed the importance threshold.

    Notes:
        - The `feature_importance` DataFrame is expected to have feature names as its index.
        - This function is used as part of a feature selection pipeline.
    """
    selected_features = feature_importance[
        feature_importance["importance"] >= threshold
    ]
    return selected_features.index.tolist()
