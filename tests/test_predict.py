from jcg_testdatascience_1.predict import make_prediction


def test_make_prediction():
    input_data = {
        "id": [0],
        "age": [63],
        "sex": ["Male"],
        "dataset": ["Cleveland"],
        "cp": ["typical angina"],
        "trestbps": [145.0],
        "chol": [233.0],
        "fbs": [True],
        "restecg": ["lv hypertrophy"],
        "thalch": [150.0],
        "exang": [False],
        "oldpeak": [2.3],
        "slope": ["downsloping"],
        "ca": [0.0],
        "thal": ["fixed defect"],
    }

    result = make_prediction(input_data=input_data)
    keys = result.keys()

    assert isinstance(result, dict)
    assert "predictions" in keys
    assert "version" in keys
    assert "errors" in keys
    assert result["errors"] is None
    assert isinstance(result["version"], str)
    assert isinstance(result["predictions"], list)
    assert result["predictions"][0] == 0
