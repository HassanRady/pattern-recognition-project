import pandas as pd

from src import utils


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df = _df.dropna(subset=utils.constants.TARGET_COLUMN_NAME)
    _df = _df.drop(columns=utils.constants.FEATURES_TO_DROP)
    return _df
