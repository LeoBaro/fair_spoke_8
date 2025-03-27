import pytest
import logging
import ray
import torch
from transformers import CLIPModel, CLIPProcessor

from made.data_pipeline.steps.multimodal_filtering import (
    multimodal_filtering, 
    MultimodalFilter
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_multimodal_filtering(tar_files, log_folder, config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(config.multimodal.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(config.multimodal.clip_model)

    results = multimodal_filtering(model, processor, tar_files, log_folder, config)
    assert len(results) == 12


def test_ray_multimodal_filtering(ray_init, ray_flag, tar_files, log_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")
    
    multimodalFilter = MultimodalFilter.remote(config_path)

    results = ray.get(
        [
            multimodalFilter.execute.remote(tar_files, log_folder)
        ]
    )

    assert len(results[0]) == 12