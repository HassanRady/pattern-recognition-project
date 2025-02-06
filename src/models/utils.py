import pandas as pd


def calculate_weights(series):
    bins = pd.cut(series, bins=10, labels=False)
    weights = bins.value_counts().reset_index()
    weights.columns = ["target_bins", "count"]
    weights["count"] = 1 / weights["count"]
    weight_map = weights.set_index("target_bins")["count"].to_dict()
    weights = bins.map(weight_map)
    return weights / weights.mean()
