from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import numpy as np
import optuna
import pandas as pd

import utils.constants
from evaluator import evaluate
from src.logger import (
    get_console_logger,
)
from utils.registry import RegressionEstimator

LOGGER = get_console_logger(logger_name=__name__)


def score_estimator(
    x: np.ndarray,
    y: np.ndarray,
    estimator: RegressionEstimator,
) -> pd.DataFrame:
    y_pred = estimator.predict(x)
    return evaluate(y_true=y, y_pred=y_pred)


def predict(estimator: RegressionEstimator, x: np.ndarray) -> pd.DataFrame:
    preds = estimator.predict(x)
    preds = preds.round(0).astype(int)
    df = pd.DataFrame(preds, columns=[utils.constants.TARGET_COLUMN_NAME])
    return df


def run_hpo(
    n_trials: int,
    hpo_path: Path,
    study_name: str,
    objective: Callable[
        ...,
        float,
    ],
    n_jobs: Optional[int] = -1,
) -> Tuple[int, dict[str, Any], float, dict[str, Any]]:
    hpo_path.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{hpo_path}/{study_name}.sqlite3",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    return (
        study.best_trial.number,
        study.best_trial.params,
        study.best_trial.value,
        study.best_trial.user_attrs,
    )
