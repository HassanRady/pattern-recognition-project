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
