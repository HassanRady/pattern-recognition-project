from enum import Enum
from typing import Union

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from sklearn.impute import KNNImputer, SimpleImputer

from torch import nn

ActivationLayerType = Union[nn.ReLU, nn.Tanh]


class ActivationLayers(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"


activation_layer_registry = {
    ActivationLayers.RELU.value: nn.ReLU,
    ActivationLayers.TANH.value: nn.Tanh,
    ActivationLayers.LEAKY_RELU.value: nn.LeakyReLU,
}

ScalerType = Union[
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
]


class SklearnScalers(Enum):
    MINMAX = "MinMaxScaler"
    STANDARD = "StandardScaler"
    MAXABS = "MaxAbsScaler"
    ROBUST = "RobustScaler"
    QUANTILE = "QuantileTransformer"
    POWER = "PowerTransformer"


sklearn_scaler_registry = {
    SklearnScalers.MINMAX.value: MinMaxScaler,
    SklearnScalers.STANDARD.value: StandardScaler,
    SklearnScalers.MAXABS.value: MaxAbsScaler,
    SklearnScalers.ROBUST.value: RobustScaler,
    SklearnScalers.QUANTILE.value: QuantileTransformer,
    SklearnScalers.POWER.value: PowerTransformer,
}


class ImputersAndInterplations(Enum):
    KNN = "KNNImputer"
    MEAN = "SimpleImputer"
    INTERPOLATION = "InterpolationTransformer"


RegressionEstimator = Union[
    SVR,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    DecisionTreeRegressor,
    KNeighborsRegressor,
    XGBRegressor,
    LGBMRegressor,
    CatBoostRegressor,
    Pipeline,
    MLPRegressor,
]

sklearn_regression_estimators_registry = {
    "svm": SVR,
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "random_forest": RandomForestRegressor,
    "ada_boost": AdaBoostRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgb": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
    "decision_tree": DecisionTreeRegressor,
    "knn": KNeighborsRegressor,
    "mlp": MLPRegressor,
}


def parse_sklearn_scaler(scaler: str) -> ScalerType:
    return sklearn_scaler_registry[scaler]


def get_estimator_importance_attribute(estimator: type[RegressionEstimator]) -> str:
    return (
        "feature_importances_"
        if hasattr(estimator, "feature_importances_")
        else "coef_"
    )
