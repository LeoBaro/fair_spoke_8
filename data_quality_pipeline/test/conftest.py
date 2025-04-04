import logging
from datetime import datetime
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
    log_folder = Path(__file__).parent / "logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder

@pytest.fixture(scope="session")
def tar_files(data_path):
    return sorted([str(data_path/s) for s in Path(data_path).glob("*.tar")])

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
def config_path(request):
    config_file_for_tests_path = "/tmp/test.yaml"
    with open(config_file_for_tests_path, "w") as f:
        f.write("""
infrastructure:
    num_workers: 2
    enable_metrics: true
    logging_level: DEBUG
    save_npy: true
    apply_filters: false

unimodal:
    batch_size: 250

    caption_min_words: 2
    caption_min_chars: 5

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    tagging_model_name: en_core_web_trf
    good_captions_pos_distribution_path: models/common_pos_patterns.txt

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 1.8

multimodal:
    batch_size: 256
                
    clip_model: openai/clip-vit-base-patch32
    clip_score_threshold: 0.3

""")
    return config_file_for_tests_path


@pytest.fixture(scope="session")
def config(config_path):
    return Config(config_path)

