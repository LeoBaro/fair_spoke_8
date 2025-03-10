import logging
from pathlib import Path

import pytest
import ray


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

@pytest.fixture(scope="session")
def ray_flag(request):
    return request.config.getoption("--ray")

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