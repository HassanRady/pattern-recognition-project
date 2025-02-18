from functools import partial
from pathlib import Path
from typing import Callable, Any

import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from src import utils
from src.features import rfe
from src.logger import get_console_logger
from src.models.registry import (
    RegressionEstimator,
    get_estimator_importance_attribute,
)

LOGGER = get_console_logger(logger_name=__name__)


def rfecv_train_hpo_objective(
    df: pd.DataFrame,
    hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    estimator: type[RegressionEstimator],
    estimator_save_path: Path,
) -> Callable[[optuna.Trial], float]:
    """
    Creates an Optuna objective function for hyperparameter optimization (HPO) that leverages
    Recursive Feature Elimination with Cross-Validation (RFECV) for feature selection.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and target data.
    target : str
        The name of the target variable in the DataFrame.
    hpo_space : Callable[[optuna.Trial], dict[str, Any]]
        A callable that defines the hyperparameter search space for Optuna trials.
    estimator : type[RegressionEstimatorType]
        The regression estimator class (sklearn estimators) to use in the pipeline.
    estimator_save_path : Path
        The directory path where the trained estimator and selected features will be saved.

    Returns:
    -------
    Callable[[optuna.Trial], float]
        A partial function that serves as the Optuna objective function. It returns the
        absolute performance score for a given trial.

    Notes:
    ------
    - Uses `rfe.get_features` to perform the feature selection and scoring, and stores
      selected features as trial attributes.
    - The performance score is minimized (via absolute value) for optimization purposes.
    """

    def _rfecv_train_hpo_objective(
        trial: optuna.Trial,
        df: pd.DataFrame,
        estimator: type[RegressionEstimator],
        estimator_save_path: Path,
        hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    ) -> float:
        """
        The inner function defining the logic of the objective function for a single Optuna trial.
        """
        params = hpo_space(trial)
        imputer_or_interpolation = params.pop("imputer_or_interpolation")
        scaler = params.pop("scaler")

        importance_getter = (
            f"named_steps.estimator.{get_estimator_importance_attribute(estimator)}"
        )

        df_reset_index = df.reset_index(drop=True)
        x = df_reset_index.drop(columns=[utils.constants.SII_COLUMN_NAME])
        y = df_reset_index[utils.constants.SII_COLUMN_NAME].astype(int)

        pipe = Pipeline(
            [
                ("imputer_or_interpolation", imputer_or_interpolation),
                ("scaler", scaler()),
                (
                    "estimator",
                    estimator(**params),
                ),
            ]
        )
        features, score = rfe.get_features(
            x=x,
            y=y,
            estimator=pipe,
            importance_getter=importance_getter,
            estimator_save_path=estimator_save_path / f"hpo_trial_{trial.number}",
            verbose=1,
        )
        trial.set_user_attr("features", features)
        return abs(score)

    return partial(
        _rfecv_train_hpo_objective,
        df=df,
        hpo_space=hpo_space,
        estimator=estimator,
        estimator_save_path=estimator_save_path,
    )
