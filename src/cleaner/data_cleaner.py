import numpy as np
import pandas as pd

import utils


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=utils.constants.TARGET_COLUMN_NAME)

    if np.any(np.isinf(df)):
        df = df.replace([np.inf, -np.inf], np.nan)

    return df
