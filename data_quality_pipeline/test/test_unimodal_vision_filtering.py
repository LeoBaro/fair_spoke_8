import pytest

import ray

from made.data_pipeline.steps.unimodal_vision_filtering import (
    unimodal_vision_filtering, 
    ray_unimodal_vision_filtering
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_unimodal_vision_filtering(tar_files):
    results = unimodal_vision_filtering(tar_files)
    MetricsStore().save_to_file(Config().infrastructure.log_folder)

@pytest.mark.skip("Skipping unimodal vision filtering")
def test_ray_unimodal_vision_filtering(ray_init, tar_files):
    config = Config()
    config.infrastructure.num_nodes = 1
    config.infrastructure.num_workers = 2
    results = ray.get([ray_unimodal_vision_filtering.remote(shards) for shards in tar_files if shards])