import pandas as pd

from data.data_manager import read_csv
from src.cleaner.data_cleaner import clean_data
from src.config import init_pipeline_config
from src.data.data_manager import (
    prepare_data_for_estimator,
    load_model,
    save_scores,
    save_csv,
)
from src.data.dataset import read_tabular_dataset
from src.logger import get_console_logger
from src.models.core import run_hpo, score_estimator, predict
from src.models.hpo_spaces import estimators_hpo_space_mapping
from src.pipeline.steps import rfecv_train_hpo_objective, subset_of_features
from src.utils.args import parse_config_path_args
from src.utils.registry import sklearn_regression_estimators_registry
from src import utils
from src.features import literature

LOGGER = get_console_logger(logger_name=__name__)


def main(config):
    LOGGER.info("Start pipeline")
    train_df, test_df = read_tabular_dataset(config.tabular_dataset_path)
    train_df = clean_data(train_df)
    target_df = train_df[utils.constants.TARGET_COLUMN_NAME]

    x_df = literature.add_features(train_df).iloc[:, -4:]
    test_df = literature.add_features(test_df).iloc[:, -4:]
    train_df = pd.merge(x_df, target_df, left_index=True, right_index=True)

    time_series_encoded_df = read_csv(config.encoded_dataset_path)

    val_scores_dfs = []
    for estimator_config in config.estimators:
        LOGGER.info(f"Start HPO for estimator: {estimator_config.name}")
        estimator = sklearn_regression_estimators_registry[estimator_config.name]
        estimator_path = config.artifacts_path / "estimators" / estimator_config.name
        best_trial_number, best_space, best_score, user_attrs = run_hpo(
            n_trials=estimator_config.hpo_trials,
            hpo_path=config.artifacts_path / "hpo" / estimator_config.name,
            study_name=config.hpo_study_name + "__" + estimator_config.name,
            objective=rfecv_train_hpo_objective(
                df=train_df,
                hpo_space=estimators_hpo_space_mapping[estimator_config.name],
                estimator=estimator,
                estimator_save_path=estimator_path,
            ),
        )

        val_scores_dfs.append(
            pd.DataFrame([best_score], columns=[estimator_config.name])
        )
        rfe_selected_features = user_attrs[utils.constants.FEATURES_COLUMN_NAME]
        x_train_df_for_estimator = subset_of_features(train_df, rfe_selected_features)
        train_df_for_estimator = pd.merge(
            x_train_df_for_estimator, target_df, left_index=True, right_index=True
        )
        x_test_df_for_estimator = subset_of_features(test_df, rfe_selected_features)

        estimator_path = estimator_path / f"hpo_trial_{best_trial_number}"
        estimator = load_model(estimator_path)

        x, y = prepare_data_for_estimator(train_df_for_estimator)
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
    val_score_df = pd.concat(val_scores_dfs, axis=1)
    save_csv(
        val_score_df,
        config.artifacts_path / "scores" / "val.csv",
    )


if __name__ == "__main__":
    args = parse_config_path_args()
    config = init_pipeline_config(args.config_path)
    main(config)
