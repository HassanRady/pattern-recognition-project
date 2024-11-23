from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict

from logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


class AutoencoderHPOConfig(BaseModel):
    """Configuration for hyperparameter optimization of a single estimator"""

    model_config = ConfigDict(protected_namespaces=())

    study_name: Optional[str] = "hpo_study"
    n_trials: Optional[int] = 1
    hpo_path: Path
    dataset_path: Path
    save_data_path: Path



def _load_config(path: Union[str, Path]) -> dict:
    """
    Loads configuration settings from a YAML file.

    This function reads a YAML file from the specified path and returns its contents
    as a dictionary.

    Parameters:
        path (Union[str, Path]): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing configuration settings from the YAML file.
    """
    LOGGER.info(f"Loading config at {path}")
    with open(path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def init_autoencoder_hpo_config(path: Union[str, Path]) -> AutoencoderHPOConfig:
    return AutoencoderHPOConfig(**_load_config(path))