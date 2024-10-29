from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from strictyaml import YAML, load

import jcg_testdatascience_1

# Directorios y ubicaciones
PACKAGE_ROOT = Path(jcg_testdatascience_1.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"


class AppConfig(BaseModel):
    """
    General-ish configurations for the App.
    """

    package_name: str
    training_data: str
    pipeline_name: str
    pipeline_save_file: str


class PipelineConfig(BaseModel):
    """
    Model-sprecific configurations.
    """

    target: str
    features: List[str]
    vars_to_drop: List[str]
    categorical_vars: List[str]
    categorical_vars_to_encode: List[str]
    categorical_vars_with_missing: List[str]
    categorical_vars_to_inpute_with_most_freq: List[str]
    categorical_vars_to_inpute_with_new_label: List[str]
    numerical_vars: List[str]
    numerical_vars_with_missing: List[str]
    numerical_vars_with_missing_mean_inputation: List[str]
    numerical_vars_with_missing_median_inputation: List[str]
    numerical_vars_to_replace_ceros: List[str]
    knn_n_neighbors: int
    knn_weights: str


class Config(BaseModel):
    """Object with all configurations groupes."""

    app_config: AppConfig
    pipeline_config: PipelineConfig


def find_config_file() -> Path:
    """Retrieves the config filepath."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parses the configuration YAML file."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Validates configurations."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        pipeline_config=PipelineConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
