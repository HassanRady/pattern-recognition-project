import os

import pandas as pd

from src.data.data_preprocessor import preprocess_data
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

    train_df, test_df = preprocess_data(config.dataset, save_path=config.artifacts_path)

    for estimator_config in config.estimators:
        estimator = sklearn_regression_estimators_registry[estimator_config.name]
        estimator_path = config.artifacts_path / "estimators" / estimator_config.name
        hpo_space = estimators_hpo_space_mapping[estimator_config.name]
        run_hpo_pipeline(
            hpo_study_name=config.hpo_study_name,
            hpo_path=config.artifacts_path / "hpo" / estimator_config.name,
            df=train_df,
            test_df=test_df,
            estimator=estimator,
            hpo_space=hpo_space,
            n_trials=estimator_config.hpo_trials,
            artifacts_path=config.artifacts_path,
        )
