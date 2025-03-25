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
        f.write(f"""
infrastructure:
  num_workers: 2
  enable_metrics: true
  logging_level: DEBUG
  save_npy: true
  apply_filters: false

unimodal:
  # ['same_input',  'classic']
  batch_size: 4000

  caption_min_words: 2
  caption_max_length: 100
  caption_max_chars: 5

  lang_detection_model_path: models/lid.176.bin
  lang_detection_score_threshold: 0.7
  lang_detection_language: en

  tagging_model_path: en_core_web_trf
  good_captions_pos_distribution_path: models/common_pos_patterns.txt

  image_min_aspect_ratio: 0.8
  image_max_aspect_ratio: 1.8

multimodal:
  batch_size: 32
  clip_processor: "openai/clip-vit-base-patch32"
  clip_model: "openai/clip-vit-base-patch32"
  caption_generation_processor: "llava-hf/llava-v1.6-mistral-7b-hf"
  caption_generator_model: "llava-hf/llava-v1.6-mistral-7b-hf"
  diffusion_model: ""
  clip_score_threshold: 0.5

""")
    return config_file_for_tests_path

@pytest.fixture(scope="session", autouse=True)
def set_test_config(config_path):
    _ = Config(config_path)

@pytest.fixture(scope="session")
def config(set_test_config):
    return Config()

