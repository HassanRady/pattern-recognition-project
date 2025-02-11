import time
from pathlib import Path

import pandas as pd

from src.data.data_cleaner import clean_features, clean_data
from src.data.dataset import read_tabular_dataset
from src.data.utils import bin_data
from src.features import literature
from src.data.data_preprocessor import (
    merge_encoded_time_series,
    merge_pca_time_series,
    process_categorical_features,
)
from src.features.feature_selector import select_features, subset_of_features
from src.config import init_pipeline_config, DatasetConfig
from src.data.data_manager import (
    load_model,
    save_scores,
    save_csv,
)
from src.logger import get_console_logger
from src.models.core import run_hpo, score_estimator, predict
from src.models.hpo_spaces import estimators_hpo_space_mapping
from src.pipeline.steps import rfecv_train_hpo_objective
from src.utils.args import parse_config_path_args
from src.models.registry import sklearn_regressors_and_classifiers_registry
from src import utils

LOGGER = get_console_logger(logger_name=__name__)


def preprocess_data(config: DatasetConfig, save_path: Path):
    train_df, test_df = read_tabular_dataset(config.tabular_dataset_path)

    train_df, test_df = merge_pca_time_series(config, train_df, test_df, save_path)
    train_df, test_df = merge_encoded_time_series(config, train_df, test_df)

    train_df = clean_features(train_df)
    test_df = clean_features(test_df)

    train_df = literature.add_features_1(train_df)
    test_df = literature.add_features_1(test_df)

    train_df = literature.add_features_2(train_df)
    test_df = literature.add_features_2(test_df)

    train_df, test_df = bin_data(train_df, test_df, n_bins=10)

    train_df, test_df = process_categorical_features(train_df, test_df)

    train_df = clean_data(train_df)
    return train_df, test_df


def main(config):
    LOGGER.info("Start pipeline")
    start_time = time.time()

    train_df, test_df = preprocess_data(config.dataset, config.artifacts_path)
    train_df.dropna(subset=utils.constants.SII_COLUMN_NAME, inplace=True)
    train_df.drop(columns=[utils.constants.PCIAT_TOTAL_CULUMN_NAME], inplace=True)

    train_df, test_df = select_features(
        train_df, test_df, config.artifacts_path, config.correlation_threshold
    )

    for estimator_config in config.estimators:
        LOGGER.info(f"Start HPO for estimator: {estimator_config.name}")
        estimator = sklearn_regressors_and_classifiers_registry[estimator_config.name]
        estimator_path = config.artifacts_path / "estimators" / estimator_config.name
        best_trial_number, best_space, best_score, user_attrs = run_hpo(
            n_trials=estimator_config.hpo_trials,
            hpo_path=config.artifacts_path / "hpo" / estimator_config.name,
            study_name=config.hpo_study_name + "__" + estimator_config.name,
            n_jobs=-1,
            objective=rfecv_train_hpo_objective(
                df=train_df,
                hpo_space=estimators_hpo_space_mapping[estimator_config.name],
                estimator=estimator,
                estimator_save_path=estimator_path,
            ),
        )

        rfe_selected_features = user_attrs[utils.constants.FEATURES_COLUMN_NAME]

        val_scores_df = pd.DataFrame(
            data={
                utils.constants.KAPPA_COLUMN_NAME: [best_score],
                utils.constants.ESTIMATOR_COLUMN_NAME: [estimator_config.name],
                utils.constants.FEATURES_COLUMN_NAME: [str(rfe_selected_features)],
                utils.constants.PARAMS_COLUMN_NAME: [str(best_space)],
            }
        )
        train_df_for_estimator = subset_of_features(train_df, rfe_selected_features)
        x_test_df_for_estimator = subset_of_features(test_df, rfe_selected_features)

        estimator_path = estimator_path / f"hpo_trial_{best_trial_number}"
        estimator = load_model(estimator_path)

        x, y = (
            train_df_for_estimator.drop(columns=[utils.constants.SII_COLUMN_NAME]),
            train_df_for_estimator[utils.constants.SII_COLUMN_NAME],
        )
        train_scores_df = score_estimator(
            x=x,
            y=y,
            estimator=estimator,
        )
        train_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = estimator_config.name
        train_scores_df[utils.constants.PARAMS_COLUMN_NAME] = str(best_space)
        train_scores_df[utils.constants.FEATURES_COLUMN_NAME] = str(
            rfe_selected_features
        )
        save_scores(
            {
                "train": train_scores_df,
                "val": val_scores_df,
            },
            config.artifacts_path / "scores",
        )

        test_preds = predict(estimator, x_test_df_for_estimator.values)
        test_preds.index = test_df.index
        save_csv(
            test_preds,
            config.artifacts_path
            / "predictions"
            / estimator_config.name
            / "submission.csv",
        )

    LOGGER.info(f"End pipeline. Elapsed time: {(time.time() - start_time)/60:.2f} min")


if __name__ == "__main__":
    args = parse_config_path_args()
    config = init_pipeline_config(args.config_path)
    main(config)
