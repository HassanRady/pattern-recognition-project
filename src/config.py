from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.logger import get_console_logger

LOGGER = get_console_logger(logger_name=__name__)


class DatasetConfig(BaseModel):
    tabular_dataset_path: Optional[Path] = None
    train_time_series_dataset_path: Optional[Path] = None
    test_time_series_dataset_path: Optional[Path] = None
    train_time_series_encoded_dataset_path: Optional[Path] = None
    test_time_series_encoded_dataset_path: Optional[Path] = None


class AutoencoderHPOConfig(BaseModel):
    """Configuration for hyperparameter optimization of a single estimator"""

    model_config = ConfigDict(protected_namespaces=())

    study_name: Optional[str] = "hpo_study"
    n_trials: Optional[int] = 1
    hpo_path: Path
    train_dataset_path: Path
    test_dataset_path: Path
    save_data_path: Path


class EstimatorsPipeline(BaseModel):
    """Configuration for the pipeline of estimators"""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    hpo_trials: int


class PipelineConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    correlation_threshold: Optional[float] = 0.0
    estimators: list[EstimatorsPipeline]
    hpo_study_name: str
    artifacts_path: Path
    dataset: DatasetConfig
    pseudo_labeling: Optional[bool] = False


class EnsembleConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    hpo_study_name: str
    artifacts_path: Path
    dataset: DatasetConfig
    ensembles: list[EstimatorsPipeline]


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


def init_pipeline_config(path: Union[str, Path]) -> PipelineConfig:
    return PipelineConfig(**_load_config(path))


def init_ensemble_config(path: Union[str, Path]) -> EnsembleConfig:
    return EnsembleConfig(**_load_config(path))
