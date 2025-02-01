import os

import pandas as pd

from src.models.tabnet import TabNetWrapper
from src import utils
from src.data.data_cleaner import clean_data
from src.data.dataset import read_tabular_dataset
from src.features import literature
from src.config import init_pipeline_config
from src.models.hpo_spaces import estimators_hpo_space_mapping
from src.data.data_manager import read_csv, save_scores, save_csv, load_model
from src.models.core import (
    run_hpo_pipeline,
    predict,
)
from src.utils.args import parse_config_path_args
from src.models.registry import sklearn_regression_estimators_registry

if __name__ == "__main__":
    config_args = parse_config_path_args()
    config_path = (
        config_args.config_path
        if config_args.config_path
        else os.environ["CONFIG_PATH"]
    )
    config = init_pipeline_config(config_path)

    train_df, test_df = read_tabular_dataset(config.tabular_dataset_path)
    train_time_series_encoded_df = read_csv(
        config.train_time_series_encoded_dataset_path
    )
    test_time_series_encoded_df = read_csv(config.test_time_series_encoded_dataset_path)

    train_df = clean_data(train_df)

    train_df = literature.add_features_1(train_df)
    test_df = literature.add_features_1(test_df)

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

    for estimator_config in config.estimators:
        estimator = sklearn_regression_estimators_registry[estimator_config.name]
        estimator_path = config.artifacts_path / "estimators" / estimator_config.name
        hpo_space = estimators_hpo_space_mapping[estimator_config.name]
        train_scores_df, val_scores_df = run_hpo_pipeline(
            hpo_study_name=config.hpo_study_name,
            hpo_path=config.artifacts_path / "hpo" / estimator_config.name,
            df=train_df,
            estimator=estimator,
            hpo_space=hpo_space,
            n_trials=estimator_config.hpo_trials,
            estimator_path=estimator_path,
        )

        train_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = estimator_config.name
        val_scores_df[utils.constants.ESTIMATOR_COLUMN_NAME] = estimator_config.name
        save_scores(
            {
                "train": train_scores_df,
                "val": val_scores_df,
            },
            config.artifacts_path / "scores",
        )

        # There is an issue with loading TabNet model because of the way it is saved while it is loaded as joblib
        if estimator_config == "tabnet":
            estimator = TabNetWrapper()
        else:
            estimator = load_model(estimator_path)

        test_preds = predict(estimator, test_df.values)
        test_preds.index = test_df.index
        save_csv(
            test_preds,
            config.artifacts_path
            / "predictions"
            / estimator_config.name
            / "submission.csv",
        )
