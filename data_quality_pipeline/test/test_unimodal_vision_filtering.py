import pytest
import logging
import ray

from made.data_pipeline.steps.unimodal_vision_filtering import (
    unimodal_vision_filtering, 
    UnimodalVisionFilter
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_unimodal_vision_filtering(tar_files, log_folder, config):
    results = unimodal_vision_filtering(tar_files, log_folder, config)

def test_ray_unimodal_vision_filtering(ray_init, ray_flag, tar_files, log_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")
    
    unimodalVisionFilter = UnimodalVisionFilter.remote(config_path)

    # single worker test 
    results = ray.get(
        [
            unimodalVisionFilter.ray_unimodal_vision_filtering.remote(tar_files, log_folder)
        ]
    )

