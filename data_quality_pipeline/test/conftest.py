import logging
from pathlib import Path

import pytest
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
    return Path(__file__).parent / "logs"

@pytest.fixture(scope="session")
def tar_files(data_path):
    return sorted([str(data_path/s) for s in Path(data_path).glob("*.tar")])

@pytest.fixture(scope="session")
def tar_file(tar_files):
    return tar_files[0:1] 

def pytest_addoption(parser):
    parser.addoption(
        "--ray", action="store_true", default=False, help="Run Ray-based tests"
    )
    # For future multi node tests 
    parser.addoption(
        "--multinode", action="store_true", default=False, help="Run multinode Ray-based tests"
    )

@pytest.fixture(scope="session")
def ray_flag(request):
    return request.config.getoption("--ray")

# For future multi node tests 
@pytest.fixture(scope="session")
def multinode_flag(request):
    return request.config.getoption("--multinode")

@pytest.fixture(scope="session")
def ray_init(ray_flag):
    if ray_flag:
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
    if ray_flag:
        ray.shutdown()

@ray.remote
def post_mortem(x):
    x += 1
    raise Exception("An exception is raised")
    return x

@pytest.fixture(scope="session")
def config_path(request, data_path):
    config_file_for_tests_path = "/tmp/test.yaml"
    with open(config_file_for_tests_path, "w") as f:
        f.write(f"""
infrastructure:
    num_nodes: 1
    num_workers: 2 
    webdataset: {str(data_path)} 
    enable_metrics: true
    logging_level: DEBUG

unimodal:
    batch_size: 20

    caption_min_length: 10
    caption_max_length: 100

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 1.8

""")
    return config_file_for_tests_path

@pytest.fixture(scope="session", autouse=True)
def set_test_config(config_path):
    _ = Config(config_path)

