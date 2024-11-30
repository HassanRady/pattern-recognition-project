import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn import metrics
from torch.ao.nn.quantized.functional import threshold

from src import utils


def threshold_rounder(y_pred, thresholds):
    thresholds = np.sort(thresholds)
    rounded_values = np.zeros_like(y_pred, dtype=int)
    for i, threshold in enumerate(thresholds):
        rounded_values[y_pred >= threshold] = i + 1

    return rounded_values


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimize_thresholds(
    scorer,
    y_true,
    y_pred,
    initial_thresholds=np.array([0.5, 1.5, 2.5]),
):

    def _evaluate_predictions(thresholds, scorer, y_true, y_pred):
        y_pred = threshold_rounder(y_pred, thresholds)
        return -scorer(y_true, y_pred)

    result = minimize(
        fun=_evaluate_predictions,
        x0=initial_thresholds,
        args=(scorer, y_true, y_pred),
        method="Nelder-Mead",
    )

    if not result.success:
        print("Warning: Optimization did not converge. Using the best result found.")

    return result.x


def _merge_non_nan(series: pd.Series) -> pd.Series:
    return series.dropna().iloc[0] if not series.dropna().empty else np.nan


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    metric_df = pd.DataFrame()
    for metric, func in METRICS.items():
        thresholds = optimize_thresholds(func, y_true, y_pred)
        y_pred = threshold_rounder(y_pred, thresholds)
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
