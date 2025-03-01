import pytest
import logging
import ray

from made.data_pipeline.steps.unimodal_vision_filtering import (
    unimodal_vision_filtering, 
    ray_unimodal_vision_filtering
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

@pytest.mark.skip("Skipping unimodal vision filtering")
def test_unimodal_vision_filtering(tar_files):
    logger = logging.getLogger("unimodal_vision_filtering")
    results = unimodal_vision_filtering(tar_files)

def test_ray_unimodal_vision_filtering(ray_init, tar_files, log_folder):
    results = ray.get([ray_unimodal_vision_filtering.remote(shards, log_folder) for shards in tar_files if shards])

