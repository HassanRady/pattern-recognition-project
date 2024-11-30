import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

from src.evaluator import evaluate
from src.data.data_manager import save_model

import pandas as pd

from src.logger import get_console_logger
from src.data.data_manager import read_csv
import src.utils as utils
from src.utils.registry import (
    get_estimator_importance_attribute,
    RegressionEstimator,
    sklearn_regression_estimators_registry,
    sklearn_scaler_registry,
)

LOGGER = get_console_logger(logger_name=__name__)


def kappa_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -evaluate(y_true, y_pred)[utils.constants.KAPPA_COLUMN_NAME]


def get_features(
    df: pd.DataFrame,
    estimator: RegressionEstimator,
    importance_getter: str,
    estimator_save_path: Optional[Path],
    verbose: Optional[int] = 1,
) -> Tuple[List[str], float]:
    df_reset_index = df.reset_index(drop=True)
    x = df_reset_index.drop(columns=[utils.constants.TARGET_COLUMN_NAME])
    y = df_reset_index[utils.constants.TARGET_COLUMN_NAME]

    rfecv = RFECV(
        estimator=estimator,
        importance_getter=importance_getter,
        scoring=make_scorer(kappa_scorer),
        min_features_to_select=1,
        step=1,
        verbose=verbose,
        n_jobs=-1,
    )
    rfecv.fit(x, y)

    n_subsets_of_features = rfecv.cv_results_["n_features"]
    best_subset_of_feature_idx = np.argwhere(
        n_subsets_of_features == rfecv.n_features_
    ).flatten()
    best_score = rfecv.cv_results_["mean_test_score"][best_subset_of_feature_idx][0]
    best_selected_features = rfecv.get_feature_names_out().tolist()

    LOGGER.info(
        f"RFECV feature selection: {len(best_selected_features)} from {len(x.columns)}"
    )
    if estimator_save_path:
        save_model(rfecv.estimator_, estimator_save_path)
    return best_selected_features, best_score

if __name__ == "__main__":
    pass
