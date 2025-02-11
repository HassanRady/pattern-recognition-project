from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer

from src.models.utils import calculate_weights
from src.evaluator import evaluate
from src.data.data_manager import save_model


from src.logger import get_console_logger
import src.utils as utils
from src.models.registry import (
    RegressionEstimator,
)

LOGGER = get_console_logger(logger_name=__name__)


def kappa_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return evaluate(y_true, y_pred)[utils.constants.KAPPA_COLUMN_NAME]


def get_features(
    x: np.ndarray,
    y: np.ndarray,
    estimator: RegressionEstimator,
    importance_getter: str,
    estimator_save_path: Optional[Path],
    verbose: Optional[int] = 1,
) -> Tuple[List[str], float]:
    rfecv = RFECV(
        cv=4,
        estimator=estimator,
        importance_getter=importance_getter,
        scoring=make_scorer(kappa_scorer, greater_is_better=True),
        min_features_to_select=1,
        step=5,
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
        f"RFECV feature selection: {len(best_selected_features)} from {x.shape[1]} features"
    )
    if estimator_save_path:
        save_model(rfecv.estimator_, estimator_save_path)
    return best_selected_features, best_score
