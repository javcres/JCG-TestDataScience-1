import typing as t

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from jcg_testdatascience_1 import __version__ as _version
from jcg_testdatascience_1.config.core import (DATASET_DIR, TRAINED_MODEL_DIR,
                                               config)


def load_dataset() -> pd.DataFrame:
    """
    Loads the trainig data.
    """
    dataframe = pd.read_csv(DATASET_DIR / config.app_config.training_data)
    transformed = transform_dataset(dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def transform_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply initial transformations to the DataFrame.
    """
    transformed = dataframe.copy()
    transformed.drop(config.pipeline_config.vars_to_drop, axis=1, inplace=True)

    # Codificamos las variables categoricas como categoricas
    transformed[config.pipeline_config.categorical_vars] = transformed[
        config.pipeline_config.categorical_vars
    ].astype(object)

    return transformed
