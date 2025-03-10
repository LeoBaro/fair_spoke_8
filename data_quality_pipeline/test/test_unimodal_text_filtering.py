import pytest
import logging  
import ray

from made.data_pipeline.steps.unimodal_text_filtering import (
    unimodal_text_filtering, 
    ray_unimodal_text_filtering
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_unimodal_text_filtering(tar_files, log_folder):
    results = unimodal_text_filtering(tar_files, log_folder)

def test_ray_unimodal_text_filtering(ray_init, ray_flag, tar_files, log_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")    
    
    # single worker test 
    results = ray.get(
        [
            ray_unimodal_text_filtering.remote(tar_files, log_folder, config_path)
        ]
    )
