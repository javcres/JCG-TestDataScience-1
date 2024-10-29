from jcg_testdatascience_1.config.core import AppConfig, Config, PipelineConfig


def test_basic_config():
    from jcg_testdatascience_1.config.core import config

    assert isinstance(config, Config)
    assert isinstance(config.app_config, AppConfig)
    assert isinstance(config.pipeline_config, PipelineConfig)
