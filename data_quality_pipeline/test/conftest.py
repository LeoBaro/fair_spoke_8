import pytest
import logging
from pathlib import Path
from datetime import datetime

import ray

from made.config import Config


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def output_folder():
    return Path(__file__).parent / "out"

@pytest.fixture(scope="session")
def log_folder():
    log_folder = Path(__file__).parent / "logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder

@pytest.fixture(scope="session")
def webdataset_output_folder():
    webdataset_output_folder = Path(__file__).parent / "webdataset_output" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    webdataset_output_folder.mkdir(parents=True, exist_ok=True)
    return webdataset_output_folder

@pytest.fixture(scope="session")
def tar_files(data_path):
    return sorted([str(data_path/s) for s in Path(data_path).glob("*.tar")])

@pytest.fixture(scope="session")
def config_path():
    config_override_for_tests = {
        "save_bad_uids": False,
        "logging_level": "DEBUG",
        "num_workers": 2,
        "num_gpus_per_worker": 0.4,
        "batch_size": 33,
        "dump_tar_every_n_samples": 1000
    }
    return Config.create_config(config_override_for_tests, "/tmp/test.yaml")

@pytest.fixture(scope="session")
def config(config_path):
    return Config(config_path)

@pytest.fixture(scope="session")
def test_images_path_aspect_ratio():
    return Path(__file__).parent / "data" / "test_images" / "aspect_ratio"

@pytest.fixture(scope="session")
def test_images_path_text_detection():
    return Path(__file__).parent / "data" / "test_images" / "text_detection"

@pytest.fixture(scope="session")
def test_images_path_similarity():
    return Path(__file__).parent / "data" / "test_images" / "similarity"

@pytest.fixture(scope="session")
def test_images_path_specificity():
    return Path(__file__).parent / "data" / "test_images" / "specificity"

@pytest.fixture(scope="function")
def ray_init():
    ray.init(
        num_cpus=4,
        num_gpus=1,
        logging_level=logging.DEBUG,
        runtime_env={
            "env_vars": {
                "RAY_DEBUG": "1"
            }
    })
    yield
    ray.shutdown()

@ray.remote
def post_mortem(x):
    x += 1
    raise Exception("An exception is raised")
    return x