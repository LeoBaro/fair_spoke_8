import pytest
import logging  
import ray
import fasttext
from made.paths import MADE_PATH

from made.data_pipeline.steps.unimodal_text_filtering import (
    unimodal_text_filtering, 
    UnimodalTextFilter
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_unimodal_text_filtering(tar_files, log_folder):
    language_detection_model = fasttext.load_model(str(MADE_PATH / Config().unimodal.lang_detection_model_path))
    results = unimodal_text_filtering(language_detection_model, tar_files, log_folder)

def test_ray_unimodal_text_filtering(ray_init, ray_flag, tar_files, log_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")    
    
    unimodalTextFilter = UnimodalTextFilter.remote(config_path)

    # single worker test 
    results = ray.get(
        [
            unimodalTextFilter.ray_unimodal_text_filtering.remote(tar_files, log_folder)
        ]
    )
