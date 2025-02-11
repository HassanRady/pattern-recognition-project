import os

import pandas as pd
from sklearn import clone

from src.config import init_ensemble_config
from src.data.data_preprocessor import preprocess_data
from src import utils
from src.data.data_cleaner import clean_data
from src.data.dataset import read_tabular_dataset
from src.features import literature
from src.models.hpo_spaces import (
    estimators_hpo_space_mapping,
    voting_hpo_space,
    stacking_hpo_space,
)
from src.data.data_manager import read_csv, save_scores, load_model, save_csv
from src.models.core import (
    run_hpo_pipeline,
    run_hpo,
    train_cv_regressor,
    train_regressor,
    train_classifier,
    predict,
)
from src.utils.args import parse_config_path_args
from src.models.registry import (
    sklearn_regressors_and_classifiers_registry,
    is_estimator_classifier,
)
import time
from functools import partial
from typing import Any, Tuple, Callable, Optional, Union

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.evaluator import evaluate, threshold_rounder, optimize_thresholds
from src.logger import (
    get_console_logger,
)
from src.models.registry import RegressionEstimator, parse_sklearn_scaler, ScalerType
from src.utils.others import process_hpo_best_space

LOGGER = get_console_logger(logger_name=__name__)


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
    ) -> float:
        if estimators_num:
            params = hpo_space(trial, estimators_num)
        elif estimator_best_params:
            params = hpo_space(trial, estimator_best_params)
        else:
            params = hpo_space(trial)

        trial.set_user_attr(
            utils.constants.FEATURES_COLUMN_NAME,
            df.columns.drop(utils.constants.SII_COLUMN_NAME).tolist(),
        )

        pipe = Pipeline(
            [
                ("imputer_or_interpolation", params.pop("imputer_or_interpolation")),
                ("scaler", params.pop("scaler")()),
                (
                    "estimator",
                    estimator(estimators=estimators, **params),
                ),
            ]
        )

        _, val_scores_df = train_cv_regressor(
            df=df,
            estimator=pipe,
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

    train_df, test_df = preprocess_data(config.dataset, config.artifacts_path)

    estimator_best_params = {
        "lightgbm": {
            "n_jobs": -1,
            "verbose": -1,
            "learning_rate": 0.09593778691887371,
            "n_estimators": 907,
            "max_depth": 22,
            "num_leaves": 33,
            "bagging_fraction": 0.8756038641046461,
            "min_data_in_leaf": 55,
            "subsample": 0.6042493281839169,
            "colsample_bytree": 0.7336030813390204,
            "objective": "regression",
        },
        "catboost": {
            "iterations": 604,
            "depth": 6,
            "learning_rate": 0.2283441275430589,
            "subsample": 0.5204089549238112,
            "silent": True,
            "objective": "RMSE",
            "bagging_temperature": 0.2313978894670557,
            "random_strength": 3.5266989103797752,
            "min_data_in_leaf": 28,
            "l2_leaf_reg": 0.0010893452367348848,
        },
        "xgb": {
            "n_jobs": -1,
            "n_estimators": 570,
            "max_depth": 5,
            "learning_rate": 0.13870841339225293,
            "reg_alpha": 0.000630311621253957,
            "reg_lambda": 0.00840600717404905,
            "subsample": 0.7062965667873157,
            "colsample_bytree": 0.6224539775624215,
            "num_parallel_tree": 18,
            "objective": "reg:tweedie",
        },
    }

    for ensemble_config in config.ensembles:
        LOGGER.info(f"Start training ensemble {ensemble_config.name}")
        ensemble_estimator = sklearn_regressors_and_classifiers_registry[
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
                sklearn_regressors_and_classifiers_registry[estimator_name](
                    **estimator_best_params
                ),
            )
            for estimator_name, estimator_best_params in estimator_best_params.items()
        ]

        _, best_params, best_score, _ = run_hpo(
            n_trials=ensemble_config.hpo_trials,
            study_name=config.hpo_study_name,
            hpo_path=config.artifacts_path / "hpo" / ensemble_config.name,
            n_jobs=-1,
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

        if is_estimator_classifier(ensemble_estimator):
            train = train_classifier
        else:
            train = train_regressor
        estimator_path = config.artifacts_path / "estimators" / ensemble_config.name

        best_params = process_hpo_best_space(best_params, ensemble_estimator, len(base_estimators))

        pipe = Pipeline(
            [
                (
                    "imputer_or_interpolation",
                    best_params.pop("imputer_or_interpolation"),
                ),
                ("scaler", best_params.pop("scaler")()),
                (
                    "estimator",
                    ensemble_estimator(estimators=base_estimators, **best_params),
                ),
            ]
        )

        train_scores_df, thresholds = train(
            df=train_df,
            estimator=pipe,
            estimator_path=estimator_path,
        )
        train_scores_df[utils.constants.PARAMS_COLUMN_NAME] = str(best_params)

        val_scores_df = pd.DataFrame(
            data={
                utils.constants.KAPPA_COLUMN_NAME: [best_score],
                utils.constants.PARAMS_COLUMN_NAME: str(best_params),
            }
        )

        train_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = ensemble_config.name
        val_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = ensemble_config.name
        save_scores(
            {
                "train": train_scores_df,
                "val": val_scores_df,
            },
            config.artifacts_path / "scores",
        )

        estimator = load_model(estimator_path)

        test_preds = predict(estimator, test_df.values, thresholds)
        test_preds.index = test_df.index
        save_csv(
            test_preds,
            config.artifacts_path
            / "predictions"
            / ensemble_config.name
            / "submission.csv",
        )
