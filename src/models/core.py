import time
from functools import partial
from pathlib import Path
from typing import Any, Tuple, Callable, Optional, Union

import numpy as np
import optuna
import pandas as pd
from sklearn import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.models.tabnet import TabNetWrapper
from src.models.utils import calculate_weights
from src.data.data_manager import (
    save_model,
    load_model,
    save_scores,
    save_csv,
)
from src import utils
from src.evaluator import evaluate, optimize_thresholds, threshold_rounder
from src.logger import (
    get_console_logger,
)
from src.models.registry import (
    RegressionEstimator,
    parse_sklearn_scaler,
    ScalerType,
    get_estimator_name,
    Classifiers,
    is_estimator_classifier,
)
from src.utils.others import process_hpo_best_space

LOGGER = get_console_logger(logger_name=__name__)


def pseudo_labeling_regressor(
    pipe: RegressionEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    unlabeled_data: pd.DataFrame,
):
    LOGGER.info("Start pseudo labeling")
    X_unlabeled = unlabeled_data.drop(
        columns=[
            utils.constants.SII_COLUMN_NAME,
            utils.constants.PCIAT_TOTAL_CULUMN_NAME,
        ]
    )
    pseudo_preds = pipe.predict(X_unlabeled)
    n_runs = 5
    pseudo_pred_runs = np.zeros((n_runs, len(X_unlabeled)))

    for i in range(n_runs):
        pipe.fit(x_train, y_train)
        pseudo_pred_runs[i] = pipe.predict(X_unlabeled)

    pseudo_std = np.std(pseudo_pred_runs, axis=0)

    confidence_threshold = np.percentile(pseudo_std, 25)

    confident_indices = pseudo_std <= confidence_threshold
    X_pseudo = X_unlabeled[confident_indices]
    y_pseudo = np.abs(pseudo_preds[confident_indices])

    X_train_extended = pd.concat([x_train, X_pseudo], axis=0)
    y_train_extended = pd.concat(
        [y_train, pd.Series(y_pseudo, index=X_pseudo.index)], axis=0
    )

    pipe.fit(X_train_extended, y_train_extended)
    return pipe


def pseudo_labeling_classifier(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    unlabeled_data: pd.DataFrame,
):
    LOGGER.info("Start pseudo labeling")
    X_unlabeled = unlabeled_data.drop(
        columns=[
            utils.constants.SII_COLUMN_NAME,
            utils.constants.PCIAT_TOTAL_CULUMN_NAME,
        ]
    )
    pseudo_probs = model.predict_proba(X_unlabeled)

    confidence_threshold = 0.9
    pseudo_labels = np.argmax(pseudo_probs, axis=1)
    max_probs = np.max(pseudo_probs, axis=1)

    confident_indices = max_probs >= confidence_threshold
    X_pseudo = X_unlabeled[confident_indices]
    y_pseudo = pseudo_labels[confident_indices]

    X_train_extended = pd.concat([x_train, X_pseudo], axis=0)
    y_train_extended = pd.concat(
        [y_train, pd.Series(y_pseudo, index=X_pseudo.index)], axis=0
    )

    model.fit(X_train_extended, y_train_extended)
    return model


def train_regressor(
    df: pd.DataFrame,
    estimator: RegressionEstimator,
    estimator_path: Path,
    pseudo_labeling: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    LOGGER.info("Start train")
    train_time_start = time.time()

    _df = df[df[utils.constants.SII_COLUMN_NAME].notnull()]
    x_train, y_train_sii, y_train_score = (
        _df.drop(
            columns=[
                utils.constants.SII_COLUMN_NAME,
                utils.constants.PCIAT_TOTAL_CULUMN_NAME,
            ]
        ),
        _df[utils.constants.SII_COLUMN_NAME],
        _df[utils.constants.PCIAT_TOTAL_CULUMN_NAME],
    )

    estimator.fit(
        x_train,
        y_train_score,
        estimator__sample_weight=calculate_weights(y_train_score),
    )
    y_pred_train = estimator.predict(x_train)
    thresholds = optimize_thresholds(y_train_sii, y_pred_train)
    y_pred_train = threshold_rounder(y_pred_train, thresholds)
    train_scores_df = evaluate(y_true=y_train_sii, y_pred=y_pred_train)

    if pseudo_labeling:
        pseudo_labeling_regressor(
            estimator,
            x_train,
            y_train_score,
            df[df[utils.constants.SII_COLUMN_NAME].isnull()],
        )

    save_model(estimator, estimator_path)

    LOGGER.info(f"End train in {round(time.time() - train_time_start, 1)} seconds")
    return train_scores_df, thresholds


def train_classifier(
    df: pd.DataFrame,
    estimator: Classifiers,
    estimator_path: Path,
    pseudo_labeling: bool = False,
) -> Tuple[pd.DataFrame, None]:
    LOGGER.info("Start train")
    train_time_start = time.time()

    _df = df[df[utils.constants.SII_COLUMN_NAME].notnull()]
    x_train, y_train_sii = (
        _df.drop(
            columns=[
                utils.constants.SII_COLUMN_NAME,
                utils.constants.PCIAT_TOTAL_CULUMN_NAME,
            ]
        ),
        _df[utils.constants.SII_COLUMN_NAME],
    )

    estimator.fit(
        x_train,
        y_train_sii,
        estimator__sample_weight=calculate_weights(y_train_sii),
    )
    y_pred_train = estimator.predict(x_train)
    train_scores_df = evaluate(y_true=y_train_sii, y_pred=y_pred_train)

    if pseudo_labeling:
        pseudo_labeling_classifier(
            estimator,
            x_train,
            y_train_sii,
            df[df[utils.constants.SII_COLUMN_NAME].isnull()],
        )
    save_model(estimator, estimator_path)

    LOGGER.info(f"End train in {round(time.time() - train_time_start, 1)} seconds")
    return train_scores_df, None


def train_cv_regressor(
    df: pd.DataFrame,
    estimator: RegressionEstimator,
    pseudo_labeling: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Start train with CV")
    train_time_start = time.time()

    _df = df[df[utils.constants.SII_COLUMN_NAME].notnull()]

    x = _df.drop(
        columns=[
            utils.constants.SII_COLUMN_NAME,
            utils.constants.PCIAT_TOTAL_CULUMN_NAME,
        ]
    )

    train_scores_dfs = []
    val_scores_dfs = []

    n_splits = 4
    SKF = StratifiedKFold(
        shuffle=True,
        n_splits=n_splits,
    )
    for fold, (train_idx, test_idx) in enumerate(
        tqdm(
            SKF.split(x, _df[utils.constants.SII_COLUMN_NAME]),
            desc="Training Folds",
            total=n_splits,
        )
    ):
        x_train, x_val = x.iloc[train_idx], x.iloc[test_idx]
        y_train_score = _df[utils.constants.PCIAT_TOTAL_CULUMN_NAME].iloc[train_idx]
        y_train_sii = _df[utils.constants.SII_COLUMN_NAME].iloc[train_idx]
        y_val_score = _df[utils.constants.PCIAT_TOTAL_CULUMN_NAME].iloc[test_idx]
        y_val_sii = _df[utils.constants.SII_COLUMN_NAME].iloc[test_idx]

        pipe = clone(estimator)

        pipe.fit(
            x_train,
            y_train_score,
            estimator__sample_weight=calculate_weights(y_train_score),
        )
        y_pred_train = pipe.predict(x_train)
        thresholds = optimize_thresholds(y_train_sii, y_pred_train)
        y_pred_train = threshold_rounder(y_pred_train, thresholds)

        if pseudo_labeling:
            pipe = pseudo_labeling_regressor(
                pipe,
                x_train,
                y_train_score,
                df[df[utils.constants.SII_COLUMN_NAME].isnull()],
            )

        y_pred_val = pipe.predict(x_val)
        y_pred_val = threshold_rounder(y_pred_val, thresholds)

        train_scores_dfs.append(evaluate(y_true=y_train_sii, y_pred=y_pred_train))
        val_scores_dfs.append(evaluate(y_true=y_val_sii, y_pred=y_pred_val))

    LOGGER.info(f"End train in {round(time.time() - train_time_start, 1)} seconds")
    return pd.concat(train_scores_dfs), pd.concat(val_scores_dfs)


def train_cv_classifier(
    df: pd.DataFrame,
    estimator: Classifiers,
    pseudo_labeling: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Start train with CV")
    train_time_start = time.time()

    _df = df[df[utils.constants.SII_COLUMN_NAME].notnull()]

    x = _df.drop(
        columns=[
            utils.constants.SII_COLUMN_NAME,
            utils.constants.PCIAT_TOTAL_CULUMN_NAME,
        ]
    )

    train_scores_dfs = []
    val_scores_dfs = []

    n_splits = 4
    SKF = StratifiedKFold(
        shuffle=True,
        n_splits=n_splits,
    )
    for fold, (train_idx, test_idx) in enumerate(
        tqdm(
            SKF.split(x, _df[utils.constants.SII_COLUMN_NAME]),
            desc="Training Folds",
            total=n_splits,
        )
    ):
        x_train, x_val = x.iloc[train_idx], x.iloc[test_idx]
        y_train_sii = _df[utils.constants.SII_COLUMN_NAME].iloc[train_idx]
        y_val_sii = _df[utils.constants.SII_COLUMN_NAME].iloc[test_idx]

        pipe = clone(estimator)

        pipe.fit(
            x_train,
            y_train_sii,
            estimator__sample_weight=calculate_weights(y_train_sii),
        )

        y_pred_train = pipe.predict(x_train)
        train_scores_dfs.append(evaluate(y_true=y_train_sii, y_pred=y_pred_train))

        if pseudo_labeling:
            pipe = pseudo_labeling_classifier(
                pipe,
                x_train,
                y_train_sii,
                df[df[utils.constants.SII_COLUMN_NAME].isnull()],
            )

        y_pred_val = pipe.predict(x_val)
        val_scores_dfs.append(evaluate(y_true=y_val_sii, y_pred=y_pred_val))

    LOGGER.info(f"End train in {round(time.time() - train_time_start, 1)} seconds")
    return pd.concat(train_scores_dfs), pd.concat(val_scores_dfs)


def score_estimator(
    x: np.ndarray,
    y: np.ndarray,
    estimator: RegressionEstimator,
) -> pd.DataFrame:
    y_pred = estimator.predict(x)
    scores_df = evaluate(y_true=y, y_pred=y_pred)
    return scores_df


def predict(
    estimator: RegressionEstimator,
    x: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    preds = estimator.predict(x)
    if thresholds is not None:
        preds = threshold_rounder(preds, thresholds)
    df = pd.DataFrame(preds, columns=[utils.constants.SII_COLUMN_NAME])
    return df


def train_hpo_objective(
    df: pd.DataFrame,
    estimator: type[RegressionEstimator],
    hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    pseudo_labeling: bool = False,
) -> Callable[[optuna.Trial], float]:
    def _train_hpo_objective(
        trial: optuna.Trial,
        df: pd.DataFrame,
        estimator: type[RegressionEstimator],
        hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    ) -> float:
        if is_estimator_classifier(estimator):
            train_cv = train_cv_classifier
        else:
            train_cv = train_cv_regressor

        params = hpo_space(trial)

        trial.set_user_attr(
            utils.constants.FEATURES_COLUMN_NAME,
            df.columns.drop(
                utils.constants.SII_COLUMN_NAME,
                utils.constants.PCIAT_TOTAL_CULUMN_NAME,
            ).tolist(),
        )

        pipe = Pipeline(
            [
                ("imputer_or_interpolation", params.pop("imputer_or_interpolation")),
                ("scaler", params.pop("scaler")()),
                (
                    "estimator",
                    estimator(**params),
                ),
            ]
        )

        _, val_scores_df = train_cv(
            df=df,
            estimator=pipe,
            pseudo_labeling=pseudo_labeling,
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
    hpo_study_name: str,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    estimator: type[RegressionEstimator],
    hpo_space: Callable[[optuna.Trial], dict[str, Any]],
    n_trials: int,
    artifacts_path: Path,
    pseudo_labeling: bool = False,
):
    if is_estimator_classifier(estimator):
        train = train_classifier
    else:
        train = train_regressor

    estimator_name = get_estimator_name(estimator)
    estimator_path = artifacts_path / "estimators" / estimator_name

    _, best_params, best_score, _ = run_hpo(
        n_trials=n_trials,
        study_name=hpo_study_name,
        hpo_path=artifacts_path / "hpo" / estimator_name,
        n_jobs=1,
        objective=train_hpo_objective(
            df=df,
            estimator=estimator,
            hpo_space=hpo_space,
            pseudo_labeling=pseudo_labeling,
        ),
    )

    best_params = process_hpo_best_space(best_params, estimator)

    pipe = Pipeline(
        [
            ("imputer_or_interpolation", best_params.pop("imputer_or_interpolation")),
            ("scaler", best_params.pop("scaler")()),
            (
                "estimator",
                estimator(**best_params),
            ),
        ]
    )

    train_scores_df, thresholds = train(
        df=df,
        estimator=pipe,
        estimator_path=estimator_path,
        pseudo_labeling=pseudo_labeling,
    )
    train_scores_df[utils.constants.PARAMS_COLUMN_NAME] = str(best_params)

    val_scores_df = pd.DataFrame(
        data={
            utils.constants.KAPPA_COLUMN_NAME: [best_score],
            utils.constants.PARAMS_COLUMN_NAME: str(best_params),
        }
    )

    train_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = estimator_name
    val_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = estimator_name
    save_scores(
        {
            "train": train_scores_df,
            "val": val_scores_df,
        },
        artifacts_path / "scores",
    )

    # There is an issue with loading TabNet model because of the way it is saved while it is loaded as joblib
    if estimator_name == "tabnet":
        estimator = TabNetWrapper()
    else:
        estimator = load_model(estimator_path)

    test_preds = predict(estimator, test_df.values, thresholds)
    test_preds.index = test_df.index
    save_csv(
        test_preds,
        artifacts_path / "predictions" / estimator_name / "submission.csv",
    )

    return train_scores_df, val_scores_df, best_params
