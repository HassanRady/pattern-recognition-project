import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Union, Optional, List

import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

from src.data.interpolations import InterpolationTransformer
from src.models.registry import (
    ImputersAndInterplations,
    RegressionEstimator,
)
from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


def summarize_value(value) -> Union[dict, list, Any]:
    """
    Summarizes the structure and metadata of a given value.

    This function inspects the type of the input value and generates a summary
    that includes relevant metadata such as type, shape, and columns for pandas
    objects or recursively summarizes the contents of lists, tuples, and dictionaries.

    Parameters:
        value: The input value to summarize. Can be a `pd.DataFrame`, `pd.Series`,
               dictionary, list, tuple, or other types.

    Returns:
        dict | list | Any: A structured summary of the input value or the value itself
                           if it does not require special handling.

    Notes:
        - For `pd.DataFrame`, includes type, shape, and columns.
        - For `pd.Series`, includes type, shape, and name.
        - For `dict`, recursively summarizes its values.
        - For `list` or `tuple`, recursively summarizes their elements.
    """
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": value.shape,
            "columns": list(value.columns),
        }
    elif isinstance(value, pd.Series):
        return {
            "type": "Series",
            "shape": value.shape,
            "name": value.name,
        }
    elif isinstance(value, dict):
        return {k: summarize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [summarize_value(v) for v in value]
    return value


def generate_param_hash(params: dict) -> str:
    """
    Generate a unique hash string based on the provided parameters.

    This function processes the input dictionary of parameters, handling nested structures
    and special types like pandas DataFrames, and generates a hash string for consistent
    identification.

    Parameters:
        params (dict): A dictionary of parameters, which can include nested structures
                       or pandas objects.

    Returns:
        str: A unique MD5 hash string representing the parameters.

    Notes:
        - Uses `summarize_value` to preprocess the parameters, ensuring compatibility
          with JSON serialization.
        - The parameters are serialized to a JSON string with sorted keys to ensure
          hash consistency.
    """
    summarized_params = summarize_value(params)
    params_str = json.dumps(summarized_params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def skip_if_exists(
    reader: Callable[..., Any],
) -> Callable[
    [Callable[..., Union[pd.DataFrame, List[str]]]],
    Callable[..., Union[pd.DataFrame, List[str]]],
]:
    """
    Decorator to skip the execution of a function if the output already exists and the parameters match.

    This decorator checks if the output file exists and validates its associated metadata
    (including a parameter hash). If both match, the function execution is skipped, and the
    existing output is read using the provided `reader` function. Otherwise, the function
    is executed, and metadata is saved for future reference.

    Parameters:
        reader (Callable[..., Any]): A function to read the output if it already exists.

    Returns:
        Callable: A decorator for functions that return either a `pd.DataFrame` or a `list`.

    Notes:
        - The decorator uses a metadata file with the `.meta` extension to store parameter hashes.
        - Metadata is validated against the parameters provided in the function call.
        - If the metadata is missing or mismatched, the function is executed, and new metadata is saved.
    """

    def decorator(
        func: Callable[..., Union[pd.DataFrame, List[str]]],
    ) -> Callable[..., Union[pd.DataFrame, List[str]]]:
        @wraps(func)
        def check(
            save_path: Optional[Union[Path, str]] = None, *args: Any, **kwargs: Any
        ) -> Union[pd.DataFrame, List[str]]:
            if not save_path:
                return func(save_path=None, *args, **kwargs)

            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            metadata_path = save_path.with_suffix(".meta")

            param_data = {"args": args, "kwargs": kwargs}
            param_hash = generate_param_hash(param_data)

            if save_path.exists():
                if metadata_path.exists():
                    with open(metadata_path, "r") as meta_file:
                        metadata = json.load(meta_file)
                        if metadata.get("param_hash") == param_hash:
                            LOGGER.info(
                                f"Skipping {func.__name__} as {save_path} exists with the same parameters"
                            )
                            return reader(save_path)
                        else:
                            LOGGER.warning(
                                f"Metadata file {metadata_path} is not synced. Recreating it."
                            )

            result = func(save_path=save_path, *args, **kwargs)

            with open(metadata_path, "w") as meta_file:
                metadata = {
                    "param_hash": param_hash,
                    "params": summarize_value(param_data),
                }
                json.dump(metadata, meta_file, indent=4)

            return result

        return check

    return decorator


def process_hpo_best_space(
    best_space: dict[str, Any], estimator: type[RegressionEstimator]
) -> dict[str, Any]:
    imputer_or_interpolation = best_space.pop("imputer_or_interpolation")

    if imputer_or_interpolation == ImputersAndInterplations.SIMPLE.value:
        imputer_strategy = best_space.pop("imputer_strategy")
        best_space["imputer_or_interpolation"] = SimpleImputer(
            strategy=imputer_strategy
        )
    elif imputer_or_interpolation == ImputersAndInterplations.KNN.value:
        n_neighbors = best_space.pop("n_neighbors")
        weights = best_space.pop("weights")
        best_space["imputer_or_interpolation"] = KNNImputer(
            n_neighbors=n_neighbors, weights=weights
        )
    elif imputer_or_interpolation == ImputersAndInterplations.INTERPOLATION.value:
        interpolation_method = best_space.pop("interpolation_method")
        best_space["imputer_or_interpolation"] = InterpolationTransformer(
            method=interpolation_method
        )
    else:
        raise ValueError("Invalid imputer_or_interpolation")

    if estimator.__name__ == "TabNetWrapper":
        best_space["optimizer_params"] = {
            "lr": best_space.pop("lr"),
            "weight_decay": best_space.pop("weight_decay"),
        }
    return best_space
