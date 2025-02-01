import pandas as pd

from src.config import DatasetConfig
from src.data.utils import bin_data
from src.data.data_cleaner import clean_data, clean_features
from src.data.data_manager import read_csv
from src.data.dataset import read_tabular_dataset
from src.features import literature


def preprocess_data(config: DatasetConfig):
    train_df, test_df = read_tabular_dataset(config.tabular_dataset_path)
    train_time_series_encoded_df = read_csv(
        config.train_time_series_encoded_dataset_path
    )
    test_time_series_encoded_df = read_csv(config.test_time_series_encoded_dataset_path)

    train_df = pd.merge(train_df, train_time_series_encoded_df, how="left", on="id")
    test_df = pd.merge(test_df, test_time_series_encoded_df, how="left", on="id")

    train_df = clean_features(train_df)
    test_df = clean_features(test_df)

    # train_df = literature.add_features_1(train_df)
    # test_df = literature.add_features_1(test_df)

    train_df = literature.add_features_2(train_df)
    test_df = literature.add_features_2(test_df)

    columns_to_bin = [
        "PAQ_A-PAQ_A_Total",
        "BMR_norm",
        "DEE_norm",
        "GS_min",
        "GS_max",
        "BIA-BIA_FFMI",
        "BIA-BIA_BMC",
        "Physical-HeartRate",
        "BIA-BIA_ICW",
        "Fitness_Endurance-Time_Sec",
        "BIA-BIA_LDM",
        "BIA-BIA_SMM",
        "BIA-BIA_TBW",
        "DEE_BMR",
        "ICW_ECW",
    ]
    train_df, test_df = bin_data(train_df, test_df, columns_to_bin, n_bins=10)

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
