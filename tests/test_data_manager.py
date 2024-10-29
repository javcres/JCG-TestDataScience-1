import pandas as pd

from jcg_testdatascience_1.config.core import config
from jcg_testdatascience_1.processing.data_manager import load_dataset


def test_load_data():
    data = load_dataset()

    # Assert dataframe basics
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] == 920
    assert data.shape[1] == 15

    # Assert variable types
    for var in data[config.pipeline_config.categorical_vars]:
        assert data[var].dtype == "object"
    for var in data[config.pipeline_config.numerical_vars]:
        assert pd.api.types.is_any_real_numeric_dtype(data[var])
    for var in config.pipeline_config.vars_to_drop:
        assert var not in data.columns
