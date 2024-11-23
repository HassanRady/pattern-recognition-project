import numpy as np
import pandas as pd
from sklearn import metrics

from src import utils


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the quadratic weighted kappa metric.

    This function calculates the quadratic weighted kappa metric, which is a measure of
    agreement between two sets of ratings. The metric's value ranges from -1 to 1, with
    1 indicating perfect agreement and -1 indicating perfect disagreement.

    Parameters:
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted target values by the model.

    Returns:
        float: The calculated quadratic weighted kappa metric.
    """
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _merge_non_nan(series: pd.Series) -> pd.Series:
    """
    Merges non-NaN values in a Series, returning the first non-NaN value or NaN if none exist.

    This helper function is used to consolidate values in a Series by dropping NaN values
    and returning the first valid entry. If all entries are NaN, it returns NaN.

    Parameters:
        series (pd.Series): A Series potentially containing NaN values.

    Returns:
        Any: The first non-NaN value in the series, or NaN if no non-NaN values are present.
    """
    return series.dropna().iloc[0] if not series.dropna().empty else np.nan


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluates predictions against true values using predefined metrics.

    This function calculates a set of metrics to assess model performance, with each metric
    score stored in a DataFrame. The results are aggregated, rounded to three decimal places,
    and returned as a single-row DataFrame.

    Parameters:
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted target values by the model.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated metric scores, aggregated
                      and rounded to three decimal places.

    Notes:
        - Metrics are defined in the global `METRICS` dictionary, where keys are metric names
          and values are callable functions.
        - Uses `_merge_non_nan` to handle any NaN values in the aggregation process.
    """
    metric_df = pd.DataFrame()
    for metric, func in METRICS.items():
        score = func(y_true, y_pred)
        metric_df = pd.concat([metric_df, pd.DataFrame({metric: [score]})], axis=0)
    return metric_df.agg(_merge_non_nan).round(3).to_frame().T


rmse = metrics.root_mean_squared_error
mae = metrics.mean_absolute_error
r2 = metrics.r2_score

METRICS = {
    utils.constants.KAPPA_COLUMN_NAME: quadratic_weighted_kappa,
    utils.constants.RMSE_COLUMN_NAME: rmse,
    utils.constants.MAE_COLUMN_NAME: mae,
    utils.constants.R2_COLUMN_NAME: r2,
}
