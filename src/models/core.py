import time
from functools import partial
from pathlib import Path
from typing import Any, Tuple, Callable, Optional, Union

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from config import PipelineConfig
from data.data_manager import prepare_data_for_estimator, save_model, save_scores
from src import utils
from src.evaluator import evaluate
from src.logger import (
    get_console_logger,
)
from src.models.registry import RegressionEstimator, parse_sklearn_scaler, ScalerType
from utils.others import process_hpo_best_space

LOGGER = get_console_logger(logger_name=__name__)


def train(
    df: pd.DataFrame,
    estimator: type[RegressionEstimator],
    estimator_path: Path,
    imputer_or_interpolation: Any,
    scaler: Optional[Union[type[ScalerType], str]] = None,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:

    LOGGER.info("Start train")
    train_time_start = time.time()

    x_train, y_train = prepare_data_for_estimator(df)

    if type(scaler) is str:
        scaler = parse_sklearn_scaler(scaler)

    estimator = Pipeline(
        [
            ("imputer_or_interpolation", imputer_or_interpolation),
            ("scaler", scaler() if scaler else None),
            (
                "estimator",
                estimator(*args, **kwargs),
            ),
        ]
    )

    estimator.fit(x_train, y_train)
    save_model(estimator, estimator_path)

    y_pred_train = estimator.predict(x_train)
    train_scores_df = evaluate(y_true=y_train, y_pred=y_pred_train)

    LOGGER.info(f"End train in {round(time.time() - train_time_start, 1)} seconds")
    return train_scores_df


def train_cv(
    df: pd.DataFrame,
    estimator: type[RegressionEstimator],
    imputer_or_interpolation: Any,
    scaler: Optional[Union[type[ScalerType], str]] = None,
    *args: Any,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    LOGGER.info("Start train")
    train_time_start = time.time()

    x = df.drop(columns=[utils.constants.TARGET_COLUMN_NAME])
    y = df[utils.constants.TARGET_COLUMN_NAME]

    train_scores_dfs = []
    val_scores_dfs = []

    SKF = StratifiedKFold(
        shuffle=True,
    )
    for fold, (train_idx, test_idx) in enumerate(
        tqdm(SKF.split(x, y), desc="Training Folds", total=5)
    ):
        x_train, x_val = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        if type(scaler) is str:
            scaler = parse_sklearn_scaler(scaler)

        pipe = Pipeline(
            [
                ("imputer_or_interpolation", imputer_or_interpolation),
                ("scaler", scaler() if scaler else None),
                (
                    "estimator",
                    estimator(*args, **kwargs),
                ),
            ]
        )

        pipe.fit(x_train, y_train)

        y_pred_train = pipe.predict(x_train)
        y_pred_val = pipe.predict(x_val)

        train_scores_dfs.append(evaluate(y_true=y_train, y_pred=y_pred_train))
        val_scores_dfs.append(evaluate(y_true=y_val, y_pred=y_pred_val))

    LOGGER.info(f"End train in {round(time.time() - train_time_start, 1)} seconds")
    return pd.concat(train_scores_dfs), pd.concat(val_scores_dfs)


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


def train_hpo_objective(
    df: pd.DataFrame,
    estimator: type[RegressionEstimator],
    hpo_space: Callable[[optuna.Trial], dict[str, Any]],
) -> Callable[[optuna.Trial], float]:

    def _train_hpo_objective(
        trial: optuna.Trial,
        df: pd.DataFrame,
        estimator: type[RegressionEstimator],
        hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    ) -> float:
        params = hpo_space(trial)

        trial.set_user_attr(
            utils.constants.FEATURES_COLUMN_NAME,
            df.columns.drop(utils.constants.TARGET_COLUMN_NAME).tolist(),
        )

        _, val_scores_df = train_cv(
            df=df,
            estimator=estimator,
            **params,
        )
        return val_scores_df.mean().loc[utils.constants.KAPPA_COLUMN_NAME]

    return partial(
        _train_hpo_objective,
        df=df,
        estimator=estimator,
        hpo_space=hpo_space,
    )


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


def run_hpo_pipeline(
    config: PipelineConfig,
    df: pd.DataFrame,
    estimator: type[RegressionEstimator],
    hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    n_trials: int,
):

    _, best_params, best_score, _ = run_hpo(
        n_trials=n_trials,
        study_name=config.hpo_study_name,
        hpo_path=config.artifacts_path / "hpo",
        objective=train_hpo_objective(
            df=df,
            estimator=estimator,
            hpo_space=hpo_space,
        ),
    )

    best_params = process_hpo_best_space(best_params)

    train_scores_df = train(
        df=df,
        estimator=estimator,
        estimator_path=config.artifacts_path / "estimator",
        **best_params,
    )
    train_scores_df[utils.constants.PARAMS_COLUMN_NAME] = str(best_params)

    val_scores_df = pd.DataFrame(data={utils.constants.KAPPA_COLUMN_NAME: [best_score],
                                       utils.constants.PARAMS_COLUMN_NAME: str(best_params),})

    return train_scores_df, val_scores_df
