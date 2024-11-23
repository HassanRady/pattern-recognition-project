from enum import Enum
from typing import Union
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)

from torch import nn

ActivationFunctionType = Union[nn.ReLU, nn.Tanh]


class ActivationFunctions(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"


activation_layer_registry = {
    ActivationFunctions.RELU.value: nn.ReLU,
    ActivationFunctions.TANH.value: nn.Tanh,
    ActivationFunctions.LEAKY_RELU.value: nn.LeakyReLU,
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
