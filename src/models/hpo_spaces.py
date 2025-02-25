from typing import Any

import optuna
import torch
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LassoCV

from src.data.utils import Impute_With_Model
from src.data.interpolations import InterpolationTransformer
from src.models.registry import (
    sklearn_scaler_registry,
    SklearnScalers,
    ActivationLayers,
    activation_layer_registry,
    ImputersAndInterplations,
    sklearn_regressors_and_classifiers_registry,
)


def imputer_or_interpolation_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    imputer_or_interpolation_type = trial.suggest_categorical(
        "imputer_or_interpolation",
        [
            imputer_or_interpolation.value
            for imputer_or_interpolation in ImputersAndInterplations
        ],
    )

    if imputer_or_interpolation_type == "SimpleImputer":
        imputer_strategy = trial.suggest_categorical(
            "imputer_strategy", ["mean", "median", "most_frequent"]
        )
        imputer_or_interpolation = SimpleImputer(strategy=imputer_strategy)

    elif imputer_or_interpolation_type == "KNNImputer":
        n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        imputer_or_interpolation = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    elif imputer_or_interpolation_type == "InterpolationTransformer":
        method = trial.suggest_categorical(
            "interpolation_method",
            [
                "linear",
                "nearest",
                "pchip",
            ],
        )
        imputer_or_interpolation = InterpolationTransformer(method=method)
    elif imputer_or_interpolation_type == "LassoImputer":
        model = LassoCV(cv=5)
        imputer_or_interpolation = Impute_With_Model(model=model, na_frac=0.4)

    else:
        raise ValueError(
            f"Invalid imputer_or_interpolation_type: {imputer_or_interpolation_type}"
        )

    return {
        "imputer_or_interpolation": imputer_or_interpolation,
    }


def scaler_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for selecting a scaler.

    This function uses Optuna to suggest a scaler from a predefined registry of
    scikit-learn scalers. The chosen scaler is returned in a dictionary for integration
    into a larger hyperparameter optimization pipeline.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the selected scaler under the key `"scaler"`, where:
        - `scaler`: A scaler class obtained from `sklearn_scaler_registry`.

    Example:
    --------
    >>> scaler_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
    }
    """
    return {
        "scaler": sklearn_scaler_registry.get(
            trial.suggest_categorical(
                "scaler", [scaler.value for scaler in SklearnScalers]
            )
        ),
    }


def svm_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1e1, log=True),
        # Fix kernel to linear for RFE feature selection compatibility
        "kernel": trial.suggest_categorical("kernel", ["linear"]),
    }


def linear_regression_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a linear regression model.

    This function uses Optuna to suggest hyperparameters for a linear regression model,
    primarily focusing on scaling options through `scaler_hpo_space`.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a linear regression model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.

    Example:
    --------
    >>> linear_regression_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
    }


def ridge_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a Ridge regression model.

    This function uses Optuna to suggest hyperparameters for a Ridge regression model,
    including scaling and the regularization strength `alpha`.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a Ridge regression model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `alpha`: Regularization strength, sampled logarithmically between `1e-4` and `1e4`.

    Example:
    --------
    >>> ridge_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "alpha": 0.1,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
    }


def lasso_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a Lasso regression model.

    This function uses Optuna to suggest hyperparameters for a Lasso regression model,
    including scaling and the regularization strength `alpha`.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a Lasso regression model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `alpha`: Regularization strength, sampled logarithmically between `1e-4` and `1e4`.

    Example:
    --------
    >>> lasso_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "alpha": 0.1,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
    }


def elastic_net_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for an ElasticNet regression model.

    This function uses Optuna to suggest hyperparameters for an ElasticNet regression model,
    including scaling, regularization strength (`alpha`), and the L1 to L2 mixing ratio (`l1_ratio`).

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for an ElasticNet regression model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `alpha`: Regularization strength, sampled logarithmically between `1e-4` and `1e4`.
        - `l1_ratio`: The L1 to L2 mixing ratio, sampled uniformly between `0` (pure Ridge) and `1` (pure Lasso).

    Example:
    --------
    >>> elastic_net_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "alpha": 0.1,
        "l1_ratio": 0.5,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
    }


def xgb_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for an XGBoost model.

    This function uses Optuna to suggest hyperparameters for an XGBoost model,
    including scaling, tree structure parameters, and learning rate.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for an XGBoost model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `n_jobs`: Number of parallel threads, fixed to `-1` for maximum parallelism.
        - `n_estimators`: Number of boosting rounds, sampled as an integer between `100` and `1000`.
        - `max_depth`: Maximum depth of the tree, controlling the complexity of the model, sampled between `3` and `10`.
        - `learning_rate`: Step size shrinkage for updates, sampled logarithmically between `1e-5` and `0.3`.
        - `min_child_weight`: Minimum sum of instance weights (hessian) required in a child node, sampled between `1` and `10`.

    Example:
    --------
    >>> xgb_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "n_jobs": -1,
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.01,
        "min_child_weight": 5,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.3, log=True),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 1e-1),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 1e-1),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "num_parallel_tree": trial.suggest_int("num_parallel_tree", 2, 30),
        "objective": trial.suggest_categorical(
            "objective", ["reg:tweedie", "reg:pseudohubererror"]
        ),
    }


def xgb_classifier_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for an XGBoost classifier.

    This function uses Optuna to suggest hyperparameters for an XGBoost classifier,
    including scaling, tree structure parameters, regularization, and learning rate.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for an XGBoost classifier, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `n_jobs`: Number of parallel threads, fixed to `-1` for maximum parallelism.
        - `n_estimators`: Number of boosting rounds, sampled as an integer between `100` and `1000`.
        - `max_depth`: Maximum depth of the tree, controlling the complexity of the model, sampled between `3` and `10`.
        - `learning_rate`: Step size shrinkage for updates, sampled logarithmically between `1e-5` and `0.3`.
        - `min_child_weight`: Minimum sum of instance weights required in a child node, sampled between `1` and `10`.
        - `gamma`: Minimum loss reduction required to make a further partition on a leaf node, sampled between `0` and `5`.
        - `subsample`: Fraction of training samples used per boosting round, sampled between `0.5` and `1.0`.
        - `colsample_bytree`: Fraction of features used per tree, sampled between `0.5` and `1.0`.
        - `reg_alpha`: L1 regularization term on weights, sampled logarithmically between `1e-5` and `1`.
        - `reg_lambda`: L2 regularization term on weights, sampled logarithmically between `1e-5` and `1`.
        - `scale_pos_weight`: Controls balance of positive and negative weights, sampled between `0.5` and `2.0`.
        - `objective`: Specifies the loss function, chosen from standard classification objectives.
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 1),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 1),
        "objective": trial.suggest_categorical(
            "objective", ["multi:softprob", "multi:softmax"]
        ),
    }


def random_forest_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a Random Forest model.

    This function uses Optuna to suggest hyperparameters for a Random Forest model,
    including scaling, tree structure parameters, and parallelization options.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a Random Forest model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `n_jobs`: Number of parallel threads, fixed to `-1` for maximum parallelism.
        - `max_depth`: Maximum depth of the tree, sampled as an integer between `10` and `100`.
        - `n_estimators`: Number of trees in the forest, sampled as an integer between `100` and `1000`.
        - `min_samples_split`: Minimum number of samples required to split an internal node, sampled as an integer between `2` and `20`.

    Example:
    --------
    >>> random_forest_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "n_jobs": -1,
        "max_depth": 50,
        "n_estimators": 500,
        "min_samples_split": 10,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        "max_depth": trial.suggest_int("max_depth", 10, 100),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }


def mlp_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a Multi-Layer Perceptron (MLP) model.

    This function uses Optuna to suggest hyperparameters for an MLP model,
    including scaling, network architecture, and training configurations.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for an MLP model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `activation`: Activation function for the hidden layers, chosen from `["identity", "logistic", "tanh", "relu"]`.
        - `solver`: Optimization solver, chosen from `["lbfgs", "sgd", "adam"]`.
        - `batch_size`: Batch size for training, chosen from `[16, 32, 64, 128, 256]`.
        - `learning_rate`: Learning rate schedule, chosen from `["constant", "invscaling", "adaptive"]`.
        - `max_iter`: Maximum number of iterations, sampled from `[200, 1000]`.
        - `hidden_layer_sizes`: List of sizes for each hidden layer, where each layer's size is sampled from `[50, 100, 150]`.

    Example:
    --------
    >>> mlp_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "activation": "relu",
        "solver": "adam",
        "batch_size": 64,
        "learning_rate": "adaptive",
        "max_iter": 500,
        "hidden_layer_sizes": [100, 150],
    }
    """
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layer_sizes = [
        trial.suggest_categorical(f"n_units_l{i}", [50, 100, 150])
        for i in range(n_layers)
    ]
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "activation": trial.suggest_categorical(
            "activation", ["identity", "logistic", "tanh", "relu"]
        ),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", ["constant", "invscaling", "adaptive"]
        ),
        "max_iter": trial.suggest_int("max_iter", 200, 1000),
        "hidden_layer_sizes": hidden_layer_sizes,
    }


def lightgbm_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a LightGBM model.

    This function uses Optuna to suggest hyperparameters for a LightGBM model,
    including scaling, learning rate, tree structure parameters, and regularization options.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a LightGBM model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `n_jobs`: Number of parallel threads, fixed to `-1` for full parallelism.
        - `verbose`: Verbosity level, fixed to `-1` to suppress logs during training.
        - `learning_rate`: Learning rate for boosting, sampled logarithmically between `0.01` and `0.1`.
        - `n_estimators`: Number of boosting rounds, sampled as an integer between `100` and `1000`.
        - `max_depth`: Maximum depth of the tree, sampled as an integer between `10` and `100`.
        - `num_leaves`: Maximum number of leaves per tree, sampled as an integer between `10` and `150`.
        - `bagging_fraction`: Fraction of data used for bagging, sampled between `0.5` and `1.0`.
        - `min_data_in_leaf`: Minimum number of data points in a leaf, sampled as an integer between `50` and `150`.

    Example:
    --------
    >>> lightgbm_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "n_jobs": -1,
        "verbose": -1,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_depth": 50,
        "num_leaves": 75,
        "bagging_fraction": 0.8,
        "min_data_in_leaf": 100,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        "verbose": trial.suggest_categorical("verbose", [-1]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 10, 100),
        "num_leaves": trial.suggest_int("num_leaves", 10, 150),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "objective": trial.suggest_categorical(
            "objective", ["poisson", "tweedie", "regression"]
        ),
    }


def lightgbm_classifier_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a LightGBM model using Optuna.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a LightGBM model.
    """
    return {
        **imputer_or_interpolation_hpo_space(
            trial
        ),  # Ensure imputation settings are included
        **scaler_hpo_space(trial),  # Ensure scaling settings are included
        "n_jobs": trial.suggest_categorical("n_jobs", [1]),
        "verbose": trial.suggest_categorical("verbose", [-1]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "max_depth": trial.suggest_int("max_depth", 10, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 100, 300),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 150, 350),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.1),
        "objective": trial.suggest_categorical("objective", ["multiclass"]),
    }


def knn_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a k-Nearest Neighbors (k-NN) model.

    This function uses Optuna to suggest hyperparameters for a k-NN model,
    including scaling, the number of neighbors, weight function, and parallelization options.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a k-NN model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `n_jobs`: Number of parallel threads, fixed to `-1` for full parallelism.
        - `n_neighbors`: Number of neighbors to use for classification or regression, sampled between `1` and `200`.
        - `weights`: Weight function used in prediction, chosen from `["uniform", "distance"]`.

    Example:
    --------
    >>> knn_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "n_jobs": -1,
        "n_neighbors": 50,
        "weights": "distance",
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 200),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
    }


def gradient_boost_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a Gradient Boosting model.

    This function uses Optuna to suggest hyperparameters for a Gradient Boosting model,
    including scaling, the number of estimators, tree depth, and minimum sample parameters.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a Gradient Boosting model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `n_estimators`: Number of boosting rounds, sampled as an integer between `50` and `500`.
        - `max_depth`: Maximum depth of the individual trees, sampled as an integer between `3` and `10`.
        - `min_samples_split`: Minimum number of samples required to split an internal node, sampled as an integer between `2` and `10`.
        - `min_samples_leaf`: Minimum number of samples required to be at a leaf node, sampled as an integer between `1` and `10`.

    Example:
    --------
    >>> gradient_boost_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def decision_tree_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a Decision Tree model.

    This function uses Optuna to suggest hyperparameters for a Decision Tree model,
    including scaling, tree depth, and minimum sample requirements for splits and leaves.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a Decision Tree model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `max_depth`: Maximum depth of the tree, sampled as an integer between `1` and `42`.
        - `min_samples_split`: Minimum number of samples required to split an internal node, sampled as an integer between `2` and `42`.
        - `min_samples_leaf`: Minimum number of samples required to be at a leaf node, sampled as an integer between `1` and `42`.

    Example:
    --------
    >>> decision_tree_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "max_depth": trial.suggest_int("max_depth", 1, 42),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 42),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 42),
    }


def catboost_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a CatBoost model.

    This function uses Optuna to suggest hyperparameters for a CatBoost model,
    including scaling, number of iterations, tree depth, learning rate, and subsampling.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a CatBoost model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `iterations`: Number of boosting iterations, sampled as an integer between `100` and `1000`.
        - `depth`: Depth of the tree, sampled as an integer between `4` and `10`.
        - `learning_rate`: Learning rate for boosting, sampled logarithmically between `0.001` and `0.3`.
        - `subsample`: Fraction of samples used for training each tree, sampled between `0.05` and `1.0`.
        - `loss_function`: Loss function to optimize, chosen from `["RMSE"]`.
        - `silent`: Whether to suppress output during training, fixed to `True`.

    Example:
    --------
    >>> catboost_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "loss_function": "RMSE",
        "silent": True,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "silent": trial.suggest_categorical("silent", [True]),
        "loss_function": trial.suggest_categorical(
            "objective", ["Tweedie:variance_power=1.5", "Poisson", "RMSE"]
        ),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 60),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 1e-1),
    }


def catboost_classifier_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    """
    Defines the hyperparameter optimization (HPO) space for a CatBoost model.

    This function uses Optuna to suggest hyperparameters for a CatBoost model,
    including scaling, number of iterations, tree depth, learning rate, and subsampling.

    Parameters:
    ----------
    trial : optuna.Trial
        An Optuna trial object used to sample hyperparameters.

    Returns:
    -------
    dict[str, Any]
        A dictionary containing the hyperparameters for a CatBoost model, including:
        - `scaler`: Scaling parameters from `scaler_hpo_space`.
        - `iterations`: Number of boosting iterations, sampled as an integer between `100` and `1000`.
        - `depth`: Depth of the tree, sampled as an integer between `4` and `10`.
        - `learning_rate`: Learning rate for boosting, sampled logarithmically between `0.001` and `0.3`.
        - `subsample`: Fraction of samples used for training each tree, sampled between `0.05` and `1.0`.
        - `loss_function`: Loss function to optimize, chosen from `["RMSE"]`.
        - `silent`: Whether to suppress output during training, fixed to `True`.

    Example:
    --------
    >>> catboost_hpo_space(trial)
    {
        "scaler": <ScalerClass>,
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "loss_function": "RMSE",
        "silent": True,
    }
    """
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "silent": trial.suggest_categorical("silent", [True]),
        "loss_function": trial.suggest_categorical("objective", ["MultiClass"]),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 60),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 1e-1),
    }


def extra_trees_classifier_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def autoencoder_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    n_layers = trial.suggest_int("n_layers", 1, 6)
    return {
        **scaler_hpo_space(trial),
        "hidden_dims": [
            trial.suggest_int(f"n_units_l{i}", 16, 1024) for i in range(n_layers)
        ],
        "activation_layer": activation_layer_registry[
            trial.suggest_categorical(
                "activation_layer",
                [activation.value for activation in ActivationLayers],
            )
        ],
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
        "epochs": trial.suggest_int("epochs", 10, 100),
        "batch_size": trial.suggest_categorical("batch_size", [128]),
    }


def tabnet_hpo_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "n_d": trial.suggest_int("n_d", 8, 128, step=8),
        "n_a": trial.suggest_int("n_a", 8, 128, step=8),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True),
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        },
        "mask_type": trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
        "scheduler_params": {
            "mode": "min",
            "patience": 10,
            "min_lr": 1e-5,
            "factor": 0.5,
        },
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "verbose": 1,
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
    }


def voting_hpo_space(trial: optuna.Trial, estimators_num: int) -> dict[str, Any]:
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "weights": [
            trial.suggest_float(f"weight_{i}", 0, 1) for i in range(estimators_num)
        ],
    }


def stacking_hpo_space(
    trial: optuna.Trial, base_estimators: dict[str, dict]
) -> dict[str, Any]:
    final_estimator_name = trial.suggest_categorical(
        "final_estimator", list(base_estimators.keys())
    )
    return {
        **imputer_or_interpolation_hpo_space(trial),
        **scaler_hpo_space(trial),
        "final_estimator": sklearn_regressors_and_classifiers_registry[
            final_estimator_name
        ](**base_estimators[final_estimator_name]),
    }


estimators_hpo_space_mapping = {
    "svm": svm_hpo_space,
    "linear_regression": linear_regression_hpo_space,
    "ridge": ridge_hpo_space,
    "lasso": lasso_hpo_space,
    "elastic_net": elastic_net_hpo_space,
    "random_forest": random_forest_hpo_space,
    "gradient_boosting": gradient_boost_hpo_space,
    "xgb": xgb_hpo_space,
    "lightgbm": lightgbm_hpo_space,
    "catboost": catboost_hpo_space,
    "decision_tree": decision_tree_hpo_space,
    "knn": knn_hpo_space,
    "mlp": mlp_hpo_space,
    "tabnet": tabnet_hpo_space,
    "voting": voting_hpo_space,
    "stacking": stacking_hpo_space,
    "xgb_classifier": xgb_classifier_hpo_space,
    "lightgbm_classifier": lightgbm_classifier_hpo_space,
    "catboost_classifier": catboost_classifier_hpo_space,
    "extra_trees_classifier": extra_trees_classifier_hpo_space,
}
