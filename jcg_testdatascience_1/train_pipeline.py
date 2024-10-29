from jcg_testdatascience_1.config.core import config
from jcg_testdatascience_1.pipeline import heart_disease_pipe
from jcg_testdatascience_1.processing.data_manager import (load_dataset,
                                                           save_pipeline)


def run_training() -> None:
    """Train the model."""
    data = load_dataset()

    X = data.drop(config.pipeline_config.target, axis=1)
    y = data[config.pipeline_config.target]

    heart_disease_pipe.fit(X, y)

    save_pipeline(pipeline_to_persist=heart_disease_pipe)


if __name__ == "__main__":
    run_training()
