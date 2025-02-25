import pytest

import ray

from made.data_pipeline.steps.unimodal_text_filtering import (
    unimodal_text_filtering, 
    ray_unimodal_text_filtering
)
from made.data_pipeline.config import Config

def test_unimodal_text_filtering(tar_files):
    results = unimodal_text_filtering(tar_files)
    
@pytest.mark.skip("Skipping unimodal text filtering")
def test_ray_unimodal_text_filtering(ray_init, tar_files):
    config = Config()
    config.infrastructure.num_nodes = 1
    config.infrastructure.num_workers = 2
    results = ray.get([ray_unimodal_text_filtering.remote(shards) for shards in tar_files if shards])