import typing as t

import pandas as pd

from jcg_testdatascience_1 import __version__ as _version
from jcg_testdatascience_1.config.core import config
from jcg_testdatascience_1.processing.data_manager import load_pipeline
from jcg_testdatascience_1.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
trained_heart_disease_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using the saved pipeline and new data."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        validated_data = validated_data[config.pipeline_config.features]
        predictions = trained_heart_disease_pipe.predict(X=validated_data)
        results = {
            "predictions": predictions.tolist(),
            "version": _version,
            "errors": errors,
        }

    return results
