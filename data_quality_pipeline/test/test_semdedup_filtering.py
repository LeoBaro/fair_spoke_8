import pytest
import logging
import ray
from transformers import CLIPModel, CLIPImageProcessor
from made.paths import MADE_PATH

from made.data_pipeline.steps.semdedup_filtering import (
    semdedup_filtering,
    SemDeDupFilter
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_semdedup_filtering(tar_files, log_folder, config):
    model = CLIPModel.from_pretrained(
        config.unimodal.semdedup.clip_model
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        config.unimodal.semdedup.clip_model
    )
    results = semdedup_filtering(
        model,
        image_processor,
        tar_files,
        log_folder,
        config
    )
    
    # Add an assertion to verify the results
    assert isinstance(results, list)
    
def test_ray_semdedup_filtering(ray_init, ray_flag, tar_files, log_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")    
    
    semDeDupFilter = SemDeDupFilter.remote(config_path)

    results = ray.get(
        [
            semDeDupFilter.execute.remote(tar_files, log_folder)
        ]
    )
    assert isinstance(results[0], list)