from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import DatasetConfig
from src.data.utils import bin_data
from src.data.data_cleaner import clean_data, clean_features
from src.data.data_manager import read_csv, save_csv
from src.data.dataset import read_tabular_dataset, load_time_series
from src.features import literature


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

    train_df = clean_data(train_df)

    train_df, test_df = process_categorical_features(train_df, test_df)

    return train_df, test_df


def process_categorical_features(train_df, test_df):
    cat_c = [
        "Basic_Demos-Enroll_Season",
        "CGAS-Season",
        "Physical-Season",
        "Fitness_Endurance-Season",
        "FGC-Season",
        "BIA-Season",
        "PAQ_A-Season",
        "PAQ_C-Season",
        "SDS-Season",
        "PreInt_EduHx-Season",
    ]

    for col in cat_c:
        a_map = {}
        all_unique = set(train_df[col].unique()) | set(test_df[col].unique())
        for i, value in enumerate(all_unique):
            a_map[value] = i

        train_df[col] = train_df[col].map(a_map)
        test_df[col] = test_df[col].map(a_map)
    return train_df, test_df


def process_categorical_features_2(df):
    _df = df.copy()

    cat_c = [
        "Basic_Demos-Enroll_Season",
        "CGAS-Season",
        "Physical-Season",
        "Fitness_Endurance-Season",
        "FGC-Season",
        "BIA-Season",
        "PAQ_A-Season",
        "PAQ_C-Season",
        "SDS-Season",
        "PreInt_EduHx-Season",
    ]

    def update(_df):
        for c in cat_c:
            _df[c] = _df[c].fillna("Missing")
            _df[c] = _df[c].astype("category")
        return _df

    def create_mapping(column, dataset):
        unique_values = dataset[column].unique()
        return {value: idx for idx, value in enumerate(unique_values)}

    for col in cat_c:
        _df[col] = _df[col].fillna("Missing")
        _df[col] = _df[col].astype("category")
        mapping = create_mapping(col, _df)
        _df[col] = _df[col].replace(mapping).astype(int)


def merge_pca_time_series(config, train_df, test_df, save_path):
    def perform_pca(train, test, n_components=None, random_state=42):
        scaler = StandardScaler()
        train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        test = pd.DataFrame(scaler.transform(test), columns=test.columns)

        for c in train.columns:
            m = np.mean(train[c])
            train[c].fillna(m, inplace=True)
            test[c].fillna(m, inplace=True)

        pca = PCA(n_components=n_components, random_state=random_state)
        train_pca = pca.fit_transform(train)
        test_pca = pca.transform(test)

        train_pca_df = pd.DataFrame(
            train_pca, columns=[f"PC_{i + 1}" for i in range(train_pca.shape[1])]
        )
        test_pca_df = pd.DataFrame(
            test_pca, columns=[f"PC_{i + 1}" for i in range(test_pca.shape[1])]
        )

        return train_pca_df, test_pca_df, pca

    if (save_path / "train_pca.csv").exists() and (save_path / "test_pca.csv").exists():
        df_train_pca = read_csv(save_path / "train_pca.csv")
        df_test_pca = read_csv(save_path / "test_pca.csv")
        train_df = pd.merge(train_df, df_train_pca, how="left", on="id")
        test_df = pd.merge(test_df, df_test_pca, how="left", on="id")
        return train_df, test_df

    train_time_series_raw_df = load_time_series(config.train_time_series_dataset_path)
    test_time_series_raw_df = load_time_series(config.test_time_series_dataset_path)

    df_train_pca, df_test_pca, pca = perform_pca(
        train=train_time_series_raw_df.drop("id", axis=1),
        test=test_time_series_raw_df.drop("id", axis=1),
        n_components=15,
    )

    df_train_pca.index = train_time_series_raw_df["id"]
    df_test_pca.index = test_time_series_raw_df["id"]

    train_df = pd.merge(train_df, df_train_pca, how="left", on="id")
    test_df = pd.merge(test_df, df_test_pca, how="left", on="id")

    if save_path:
        save_csv(df_train_pca, save_path / "train_pca.csv")
        save_csv(df_test_pca, save_path / "test_pca.csv")

    return train_df, test_df


def merge_encoded_time_series(config, train_df, test_df):
    train_time_series_encoded_df = read_csv(
        config.train_time_series_encoded_dataset_path
    )
    test_time_series_encoded_df = read_csv(config.test_time_series_encoded_dataset_path)
    train_df = pd.merge(train_df, train_time_series_encoded_df, how="left", on="id")
    test_df = pd.merge(test_df, test_time_series_encoded_df, how="left", on="id")
    return train_df, test_df
