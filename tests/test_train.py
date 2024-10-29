import os

from jcg_testdatascience_1.config.core import TRAINED_MODEL_DIR, config
from jcg_testdatascience_1.train_pipeline import run_training


def test_model_training():
    run_training()

    files = os.listdir(TRAINED_MODEL_DIR)
    file_found = False
    # Check the trained pipeline exists in the output directory
    for file in files:
        if file.startswith(config.app_config.pipeline_save_file):
            file_found = True

    assert file_found
