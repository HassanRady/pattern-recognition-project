import numpy as np
import pandas as pd


# source: https://www.kaggle.com/code/ichigoe/lb0-494-with-tabnet
def add_features_1(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()

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


# source: https://www.kaggle.com/code/lennarthaupts/1st-place-cmi-model-v4-1-1-reduced?scriptVersionId=213769368
def add_features_2(df):
    df = df.copy()

    # From here on own features
    def assign_group(age):
        thresholds = [5, 6, 7, 8, 10, 12, 14, 17, 22]
        for i, j in enumerate(thresholds):
            if age <= j:
                return i
        return np.nan

    # Age groups
    df["group"] = df["Basic_Demos-Age"].apply(assign_group)

    # BMI
    BMI_map = {
        0: 16.3,
        1: 15.9,
        2: 16.1,
        3: 16.8,
        4: 17.3,
        5: 19.2,
        6: 20.2,
        7: 22.3,
        8: 23.6,
    }
    df["BMI_mean_norm"] = df[["Physical-BMI", "BIA-BIA_BMI"]].mean(axis=1) / df[
        "group"
    ].map(BMI_map)

    # FGC zone aggregate
    zones = [
        "FGC-FGC_CU_Zone",
        "FGC-FGC_GSND_Zone",
        "FGC-FGC_GSD_Zone",
        "FGC-FGC_PU_Zone",
        "FGC-FGC_SRL_Zone",
        "FGC-FGC_SRR_Zone",
        "FGC-FGC_TL_Zone",
    ]

    df["FGC_Zones_mean"] = df[zones].mean(axis=1)
    df["FGC_Zones_min"] = df[zones].min(axis=1)
    df["FGC_Zones_max"] = df[zones].max(axis=1)

    # Grip
    GSD_max_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 16.2, 5: 19.9, 6: 26.1, 7: 31.3, 8: 35.4}
    GSD_min_map = {0: 9, 1: 9, 2: 9, 3: 9, 4: 14.4, 5: 17.8, 6: 23.4, 7: 27.8, 8: 31.1}

    df["GS_max"] = df[["FGC-FGC_GSND", "FGC-FGC_GSD"]].max(axis=1) / df["group"].map(
        GSD_max_map
    )
    df["GS_min"] = df[["FGC-FGC_GSND", "FGC-FGC_GSD"]].min(axis=1) / df["group"].map(
        GSD_min_map
    )

    # Curl-ups, push-ups, trunk-lifts... normalized based on age-group
    cu_map = {
        0: 1.0,
        1: 3.0,
        2: 5.0,
        3: 7.0,
        4: 10.0,
        5: 14.0,
        6: 20.0,
        7: 20.0,
        8: 20.0,
    }
    pu_map = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 7.0, 6: 8.0, 7: 10.0, 8: 14.0}
    tl_map = {
        0: 8.0,
        1: 8.0,
        2: 8.0,
        3: 9.0,
        4: 9.0,
        5: 10.0,
        6: 10.0,
        7: 10.0,
        8: 10.0,
    }

    df["CU_norm"] = df["FGC-FGC_CU"] / df["group"].map(cu_map)
    df["PU_norm"] = df["FGC-FGC_PU"] / df["group"].map(pu_map)
    df["TL_norm"] = df["FGC-FGC_TL"] / df["group"].map(tl_map)

    # Reach
    df["SR_min"] = df[["FGC-FGC_SRL", "FGC-FGC_SRR"]].min(axis=1)
    df["SR_max"] = df[["FGC-FGC_SRL", "FGC-FGC_SRR"]].max(axis=1)

    # BIA Features
    # Energy Expenditure
    bmr_map = {
        0: 934.0,
        1: 941.0,
        2: 999.0,
        3: 1048.0,
        4: 1283.0,
        5: 1255.0,
        6: 1481.0,
        7: 1519.0,
        8: 1650.0,
    }
    dee_map = {
        0: 1471.0,
        1: 1508.0,
        2: 1640.0,
        3: 1735.0,
        4: 2132.0,
        5: 2121.0,
        6: 2528.0,
        7: 2566.0,
        8: 2793.0,
    }
    df["BMR_norm"] = df["BIA-BIA_BMR"] / df["group"].map(bmr_map)
    df["DEE_norm"] = df["BIA-BIA_DEE"] / df["group"].map(dee_map)
    df["DEE_BMR"] = df["BIA-BIA_DEE"] - df["BIA-BIA_BMR"]

    # FMM
    ffm_map = {
        0: 42.0,
        1: 43.0,
        2: 49.0,
        3: 54.0,
        4: 60.0,
        5: 76.0,
        6: 94.0,
        7: 104.0,
        8: 111.0,
    }
    df["FFM_norm"] = df["BIA-BIA_FFM"] / df["group"].map(ffm_map)

    # ECW ICW
    df["ICW_ECW"] = df["BIA-BIA_ECW"] / df["BIA-BIA_ICW"]

    drop_feats = [
        "FGC-FGC_GSND",
        "FGC-FGC_GSD",
        "FGC-FGC_CU_Zone",
        "FGC-FGC_GSND_Zone",
        "FGC-FGC_GSD_Zone",
        "FGC-FGC_PU_Zone",
        "FGC-FGC_SRL_Zone",
        "FGC-FGC_SRR_Zone",
        "FGC-FGC_TL_Zone",
        "Physical-BMI",
        "BIA-BIA_BMI",
        "FGC-FGC_CU",
        "FGC-FGC_PU",
        "FGC-FGC_TL",
        "FGC-FGC_SRL",
        "FGC-FGC_SRR",
        "BIA-BIA_BMR",
        "BIA-BIA_DEE",
        "BIA-BIA_Frame_num",
        "BIA-BIA_FFM",
    ]
    df = df.drop(drop_feats, axis=1)
    return df
