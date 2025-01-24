import os

import pandas as pd

from config import init_ensemble_config
from src import utils
from src.data.data_cleaner import clean_data
from src.data.dataset import read_tabular_dataset
from src.features import literature
from src.models.hpo_spaces import (
    estimators_hpo_space_mapping,
    imputer_or_interpolation_hpo_space,
    scaler_hpo_space,
    voting_hpo_space,
    stacking_hpo_space,
)
from src.data.data_manager import read_csv, save_scores
from src.models.core import (
    run_hpo_pipeline,
    run_hpo,
)
from src.utils.args import parse_config_path_args
from src.models.registry import sklearn_regression_estimators_registry
import time
from functools import partial
from typing import Any, Tuple, Callable, Optional, Union

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.evaluator import evaluate
from src.logger import (
    get_console_logger,
)
from src.models.registry import RegressionEstimator, parse_sklearn_scaler, ScalerType


LOGGER = get_console_logger(logger_name=__name__)


def ensemble_train_cv(
    df: pd.DataFrame,
    estimator: RegressionEstimator,
    imputer_or_interpolation: Any,
    scaler: Optional[Union[type[ScalerType], str]] = None,
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
                    estimator,
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


def ensemble_hpo_objective(
    df: pd.DataFrame,
    estimator: type[RegressionEstimator],
    hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    estimators: list[Tuple[str, RegressionEstimator]],
    estimators_num: Optional[int] = None,
    estimator_best_params: Optional[dict[str, dict[str, Any]]] = None,
    **kwargs: Any,
) -> Callable[[optuna.Trial], float]:
    def _ensemble_hpo_objective(
        trial: optuna.Trial,
        df: pd.DataFrame,
        estimator: type[RegressionEstimator],
        hpo_space: Callable[
            [optuna.Trial, Optional[int], Optional[dict]], dict[str, Any]
        ],
        **kwargs: Any,
    ) -> float:
        if estimators_num:
            params = hpo_space(trial, estimators_num)
            kwargs["imputer_or_interpolation"] = params.pop("imputer_or_interpolation")
            kwargs["scaler"] = params.pop("scaler")
        elif estimator_best_params:
            params = hpo_space(trial, estimator_best_params)
            kwargs["imputer_or_interpolation"] = params.pop("imputer_or_interpolation")
            kwargs["scaler"] = params.pop("scaler")

        estimator = estimator(estimators=estimators, **params)

        trial.set_user_attr(
            utils.constants.FEATURES_COLUMN_NAME,
            df.columns.drop(utils.constants.TARGET_COLUMN_NAME).tolist(),
        )

        _, val_scores_df = ensemble_train_cv(
            df=df,
            estimator=estimator,
            **kwargs,
        )
        return val_scores_df.mean().loc[utils.constants.KAPPA_COLUMN_NAME]

    return partial(
        _ensemble_hpo_objective,
        df=df,
        estimator=estimator,
        hpo_space=hpo_space,
        **kwargs,
    )


if __name__ == "__main__":
    config_args = parse_config_path_args()
    config_path = (
        config_args.config_path
        if config_args.config_path
        else os.environ["CONFIG_PATH"]
    )
    config = init_ensemble_config(config_path)

    train_df, test_df = read_tabular_dataset(config.dataset.tabular_dataset_path)
    train_time_series_encoded_df = read_csv(
        config.dataset.train_time_series_encoded_dataset_path
    )
    test_time_series_encoded_df = read_csv(config.dataset.test_time_series_encoded_dataset_path)

    train_df = clean_data(train_df)

    train_df = literature.add_features(train_df)
    test_df = literature.add_features(test_df)

    train_df = pd.merge(
        train_df, train_time_series_encoded_df, left_index=True, right_index=True
    )
    test_df = pd.merge(
        test_df,
        test_time_series_encoded_df,
        how="left",
        left_index=True,
        right_index=True,
    )

    estimators_trials = {}
    for ensemble_config in config.ensemble_estimators:
        for estimator_config in ensemble_config.estimators:
            estimators_trials[estimator_config.name] = estimator_config.hpo_trials

    estimator_best_params = {}
    for estimator_name, hpo_trials in estimators_trials.items():
        LOGGER.info(f"Start training estimator {estimator_name}")
        estimator = sklearn_regression_estimators_registry[estimator_name]
        estimator_path = config.artifacts_path / "estimators" / estimator_name
        hpo_space = estimators_hpo_space_mapping[estimator_name]
        train_scores_df, val_scores_df, best_params = run_hpo_pipeline(
            hpo_study_name=config.hpo_study_name,
            hpo_path=config.artifacts_path / "hpo" / estimator_name,
            df=train_df,
            estimator=estimator,
            hpo_space=hpo_space,
            n_trials=hpo_trials,
            estimator_path=estimator_path,
        )

        best_params.pop("imputer_or_interpolation")
        best_params.pop("scaler")
        estimator_best_params[estimator_name] = best_params

        train_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = estimator_name
        val_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = estimator_name
        save_scores(
            {
                "train": train_scores_df,
                "val": val_scores_df,
            },
            config.artifacts_path / "scores",
        )

    for ensemble_config in config.ensemble_estimators:
        LOGGER.info(f"Start training ensemble {ensemble_config.name}")
        ensemble_estimator = sklearn_regression_estimators_registry[
            ensemble_config.name
        ]
        ensemble_estimator_path = (
            config.artifacts_path / "estimators" / ensemble_config.name
        )
        if ensemble_config.name == "voting":
            hpo_space = voting_hpo_space
        elif ensemble_config.name == "stacking":
            hpo_space = stacking_hpo_space
        else:
            raise ValueError(f"Unknown ensemble estimator {ensemble_config.name}")

        base_estimators = [
            (
                estimator_name,
                sklearn_regression_estimators_registry[estimator_name](
                    **estimator_best_params
                ),
            )
            for estimator_name, estimator_best_params in estimator_best_params.items()
        ]

        _, best_params, best_score, _ = run_hpo(
            n_trials=ensemble_config.hpo_trials,
            study_name=config.hpo_study_name,
            hpo_path=config.artifacts_path / "hpo" / ensemble_config.name,
            n_jobs=1,
            objective=ensemble_hpo_objective(
                df=train_df,
                estimator=ensemble_estimator,
                estimators=base_estimators,
                hpo_space=hpo_space,
                estimators_num=(
                    len(base_estimators) if ensemble_config.name == "voting" else None
                ),
                estimator_best_params=(
                    estimator_best_params
                    if ensemble_config.name == "stacking"
                    else None
                ),
            ),
        )

        print(best_score)
