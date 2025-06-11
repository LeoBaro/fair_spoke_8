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
from made.data_pipeline.steps.base import FilteringResult

def test_multimodal_filtering(tar_files, log_folder, webdataset_output_folder, config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(config.multimodal.dfn_model).to(device)
    processor = CLIPProcessor.from_pretrained(config.multimodal.dfn_model)

    metrics_store = MetricsStore(log_folder)
    filtering_result = FilteringResult(webdataset_output_folder, config.infrastructure.dump_tar_every_n_samples)

    produced_tar_files = multimodal_filtering(
        tar_files,
        model,
        processor,
        config,
        metrics_store,
        filtering_result
    )
    assert len(produced_tar_files) == 1

def test_ray_multimodal_filtering(ray_init, ray_flag, tar_files, log_folder, webdataset_output_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")
    
    multimodalFilter = MultimodalFilter.remote(config_path, log_folder, webdataset_output_folder)

    results = ray.get(
        [
            multimodalFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files = results[0]
    assert len(produced_tar_files) == 1
