import pytest
import logging
import ray

from made.data_pipeline.steps.unimodal_vision_filtering import (
    unimodal_vision_filtering, 
    ray_unimodal_vision_filtering
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_unimodal_vision_filtering(tar_file, log_folder):
    results = unimodal_vision_filtering(tar_file, log_folder)

def test_ray_unimodal_vision_filtering(ray_init, ray_flag, tar_file, log_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")
    results = ray.get([ray_unimodal_vision_filtering.remote(shards, log_folder, config_path) for shards in tar_file if shards])

