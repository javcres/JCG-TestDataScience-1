from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from jcg_testdatascience_1.processing.data_manager import transform_dataset


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Check inputs for any errors."""

    relevant_data = transform_dataset(input_data)
    validated_data = relevant_data
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleHeartDiseaseDataInputs(
            inputs=[
                HeartDiseaseDataInputSchema(**record)
                for record in validated_data.replace({np.nan: None}).to_dict(
                    orient="records"
                )
            ]
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class HeartDiseaseDataInputSchema(BaseModel):
    """
    Class for input row validation
    """

    age: Optional[float]
    sex: Optional[str]
    dataset: Optional[str]
    cp: Optional[str]
    trestbps: Optional[float]
    chol: Optional[float]
    fbs: Optional[bool]
    restecg: Optional[str]
    thalch: Optional[float]
    exang: Optional[bool]
    oldpeak: Optional[float]
    slope: Optional[str]
    ca: Optional[float]
    thal: Optional[str]


class MultipleHeartDiseaseDataInputs(BaseModel):
    """
    Class for input validation
    """

    inputs: List[HeartDiseaseDataInputSchema]
