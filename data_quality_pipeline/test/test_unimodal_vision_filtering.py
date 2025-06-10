import pytest
import logging
import ray

from made.data_pipeline.steps.unimodal_vision_filtering import (
    unimodal_vision_filtering, 
    UnimodalVisionFilter
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
import easyocr
from made.data_pipeline.steps.base import FilteringResult

def test_unimodal_vision_filtering(tar_files, log_folder, webdataset_output_folder, config):
    
    text_detection_model = easyocr.Reader(['en'], gpu=True, user_network_directory=config.unimodal_vision.text_detection_model_path)

    metrics_store = MetricsStore(log_folder)
    filtering_result = FilteringResult(webdataset_output_folder, config.infrastructure.dump_tar_every_n_samples)

    produced_tar_files = unimodal_vision_filtering(
        tar_files,
        text_detection_model,
        config,
        metrics_store,
        filtering_result
    )
    assert len(produced_tar_files) == 1

def test_ray_unimodal_vision_filtering(ray_init, ray_flag, tar_files, log_folder, webdataset_output_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")
    
    unimodalVisionFilter = UnimodalVisionFilter.remote(config_path, log_folder, webdataset_output_folder)

    # single worker test 
    results = ray.get(
        [
            unimodalVisionFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files = results[0]
    assert len(produced_tar_files) == 1
