import pandas as pd


def simple_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values with the mean of the column.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with missing values imputed.
    """
    return df.fillna(df.mean())