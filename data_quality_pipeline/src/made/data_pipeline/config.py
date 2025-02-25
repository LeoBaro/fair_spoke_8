import yaml
from pathlib import Path
class Singleton(type):  
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DictWrapper:
    """Helper class to convert dictionary to object-like access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, DictWrapper(value) if isinstance(value, dict) else value)

class Config(metaclass=Singleton):
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)
        for key, value in self._config.items():
            setattr(self, key, DictWrapper(value) if isinstance(value, dict) else value)
