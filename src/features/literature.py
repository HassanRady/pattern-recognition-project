
import pandas as pd

import utils.constants


# source: https://www.kaggle.com/code/ichigoe/lb0-494-with-tabnet
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    featuresCols = [
        "Basic_Demos-Age",
        "Basic_Demos-Sex",
        "CGAS-CGAS_Score",
        "Physical-BMI",
        "Physical-Height",
        "Physical-Weight",
        "Physical-Waist_Circumference",
        "Physical-Diastolic_BP",
        "Physical-HeartRate",
        "Physical-Systolic_BP",
        "Fitness_Endurance-Max_Stage",
        "Fitness_Endurance-Time_Mins",
        "Fitness_Endurance-Time_Sec",
        "FGC-FGC_CU",
        "FGC-FGC_CU_Zone",
        "FGC-FGC_GSND",
        "FGC-FGC_GSND_Zone",
        "FGC-FGC_GSD",
        "FGC-FGC_GSD_Zone",
        "FGC-FGC_PU",
        "FGC-FGC_PU_Zone",
        "FGC-FGC_SRL",
        "FGC-FGC_SRL_Zone",
        "FGC-FGC_SRR",
        "FGC-FGC_SRR_Zone",
        "FGC-FGC_TL",
        "FGC-FGC_TL_Zone",
        "BIA-BIA_Activity_Level_num",
        "BIA-BIA_BMC",
        "BIA-BIA_BMI",
        "BIA-BIA_BMR",
        "BIA-BIA_DEE",
        "BIA-BIA_ECW",
        "BIA-BIA_FFM",
        "BIA-BIA_FFMI",
        "BIA-BIA_FMI",
        "BIA-BIA_Fat",
        "BIA-BIA_Frame_num",
        "BIA-BIA_ICW",
        "BIA-BIA_LDM",
        "BIA-BIA_LST",
        "BIA-BIA_SMM",
        "BIA-BIA_TBW",
        "PAQ_A-PAQ_A_Total",
        "PAQ_C-PAQ_C_Total",
        "SDS-SDS_Total_Raw",
        "SDS-SDS_Total_T",
        "PreInt_EduHx-computerinternet_hoursday",
        "BMI_Age",
        "Internet_Hours_Age",
        "BMI_Internet_Hours",
        "BFP_BMI",
        "FFMI_BFP",
        "FMI_BFP",
        "LST_TBW",
        "BFP_BMR",
        "BFP_DEE",
        "BMR_Weight",
        "DEE_Weight",
        "SMM_Height",
        "Muscle_to_Fat",
        "Hydration_Status",
        "ICW_TBW",
    ]

    season_cols = [col for col in df.columns if "Season" in col]
    x = df.drop(season_cols, axis=1)
    x["BMI_Age"] = x["Physical-BMI"] * x["Basic_Demos-Age"]
    x["Internet_Hours_Age"] = (
        x["PreInt_EduHx-computerinternet_hoursday"] * x["Basic_Demos-Age"]
    )
    x["BMI_Internet_Hours"] = (
        x["Physical-BMI"] * x["PreInt_EduHx-computerinternet_hoursday"]
    )
    x["BFP_BMI"] = x["BIA-BIA_Fat"] / x["BIA-BIA_BMI"]
    x["FFMI_BFP"] = x["BIA-BIA_FFMI"] / x["BIA-BIA_Fat"]
    x["FMI_BFP"] = x["BIA-BIA_FMI"] / x["BIA-BIA_Fat"]
    x["LST_TBW"] = x["BIA-BIA_LST"] / x["BIA-BIA_TBW"]
    x["BFP_BMR"] = x["BIA-BIA_Fat"] * x["BIA-BIA_BMR"]
    x["BFP_DEE"] = x["BIA-BIA_Fat"] * x["BIA-BIA_DEE"]
    x["BMR_Weight"] = x["BIA-BIA_BMR"] / x["Physical-Weight"]
    x["DEE_Weight"] = x["BIA-BIA_DEE"] / x["Physical-Weight"]
    x["SMM_Height"] = x["BIA-BIA_SMM"] / x["Physical-Height"]
    x["Muscle_to_Fat"] = x["BIA-BIA_SMM"] / x["BIA-BIA_FMI"]
    x["Hydration_Status"] = x["BIA-BIA_TBW"] / x["Physical-Weight"]
    x["ICW_TBW"] = x["BIA-BIA_ICW"] / x["BIA-BIA_TBW"]

    return x[featuresCols]
