import os
import yaml
from pathlib import Path

from data_quality_pipeline.src.made.data_pipeline.common import Singleton, DictWrapper

class Config(metaclass=Singleton):

    def __init__(self, config_path):
        # if config_path is None:
        #     config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)
        for key, value in self._config.items():
            setattr(self, key, DictWrapper(value) if isinstance(value, dict) else value)

    def __str__(self):
        return yaml.dump(self._config, default_flow_style=False)