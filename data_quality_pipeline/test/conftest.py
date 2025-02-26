import pytest
from pathlib import Path
import ray

@pytest.fixture(scope="session")
def ray_init():
    ray.init(
        num_cpus=4, 
        num_gpus=1,
        runtime_env={
            "env_vars": {"RAY_DEBUG": "1"},     
    })
    yield
    ray.shutdown()

@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def tar_files(data_path):
    return sorted([str(data_path/s) for s in Path(data_path).glob("*.tar")])


@ray.remote
def post_mortem(x):
    x += 1
    raise Exception("An exception is raised")
    return x