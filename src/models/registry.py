from enum import Enum
from typing import Union

from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    ExtraTreesClassifier,
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
from xgboost import XGBRegressor, XGBClassifier

from torch import nn

from src.models.tabnet import TabNetWrapper

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
    SIMPLE = "SimpleImputer"
    INTERPOLATION = "InterpolationTransformer"
    # LASSO = "LassoImputer"


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
    TabNetWrapper,
]

sklearn_regressors_and_classifiers_registry = {
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
    "tabnet": TabNetWrapper,
    "voting": VotingRegressor,
    "stacking": StackingRegressor,
    "xgb_classifier": XGBClassifier,
    "lightgbm_classifier": LGBMClassifier,
    "catboost_classifier": CatBoostClassifier,
    "extra_trees_classifier": ExtraTreesClassifier,
}

Classifiers = Union[
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier,
    ExtraTreesClassifier,
]

sklearn_classifiers_registry = {
    "xgb_classifier": XGBClassifier,
    "lightgbm_classifier": LGBMClassifier,
    "catboost_classifier": CatBoostClassifier,
    "extra_trees_classifier": ExtraTreesClassifier,
}


def get_estimator_name(estimator: type) -> str:
    """Returns the key (name) of the estimator from the registry."""
    for name, cls in sklearn_regressors_and_classifiers_registry.items():
        if cls is estimator:
            return name
    raise ValueError("Estimator not found in registry")


def parse_sklearn_scaler(scaler: str) -> ScalerType:
    return sklearn_scaler_registry[scaler]


def get_estimator_importance_attribute(estimator: type[RegressionEstimator]) -> str:
    return (
        "feature_importances_"
        if hasattr(estimator, "feature_importances_")
        else "coef_"
    )


def is_estimator_classifier(estimator: type) -> bool:
    return estimator in sklearn_classifiers_registry.values()
