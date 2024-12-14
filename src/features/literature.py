import pandas as pd


# source: https://www.kaggle.com/code/ichigoe/lb0-494-with-tabnet
def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

    _df["BMI_Age"] = _df["Physical-BMI"] * _df["Basic_Demos-Age"]
    _df["Internet_Hours_Age"] = (
        _df["PreInt_EduHx-computerinternet_hoursday"] * _df["Basic_Demos-Age"]
    )
    _df["BMI_Internet_Hours"] = (
        _df["Physical-BMI"] * _df["PreInt_EduHx-computerinternet_hoursday"]
    )
    _df["BFP_BMI"] = _df["BIA-BIA_Fat"] / _df["BIA-BIA_BMI"]
    _df["FFMI_BFP"] = _df["BIA-BIA_FFMI"] / _df["BIA-BIA_Fat"]
    _df["FMI_BFP"] = _df["BIA-BIA_FMI"] / _df["BIA-BIA_Fat"]
    _df["LST_TBW"] = _df["BIA-BIA_LST"] / _df["BIA-BIA_TBW"]
    _df["BFP_BMR"] = _df["BIA-BIA_Fat"] * _df["BIA-BIA_BMR"]
    _df["BFP_DEE"] = _df["BIA-BIA_Fat"] * _df["BIA-BIA_DEE"]
    _df["BMR_Weight"] = _df["BIA-BIA_BMR"] / _df["Physical-Weight"]
    _df["DEE_Weight"] = _df["BIA-BIA_DEE"] / _df["Physical-Weight"]
    _df["SMM_Height"] = _df["BIA-BIA_SMM"] / _df["Physical-Height"]
    _df["Muscle_to_Fat"] = _df["BIA-BIA_SMM"] / _df["BIA-BIA_FMI"]
    _df["Hydration_Status"] = _df["BIA-BIA_TBW"] / _df["Physical-Weight"]
    _df["ICW_TBW"] = _df["BIA-BIA_ICW"] / _df["BIA-BIA_TBW"]
    return _df
