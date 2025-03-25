import pytest
import logging
import ray
import fasttext
import spacy
from made.paths import MADE_PATH

from made.data_pipeline.steps.unimodal_text_filtering import (
    unimodal_text_filtering, 
    UnimodalTextFilter
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_unimodal_text_filtering(tar_files, log_folder, config):
    language_detection_model = fasttext.load_model(
        str(MADE_PATH / config.unimodal.lang_detection_model_path)
        )
    spacy.require_gpu()
    tagging_model = spacy.load(
        str(config.unimodal.tagging_model_name)
        )
    with open(
        str(MADE_PATH / config.unimodal.good_captions_pos_distribution_path),
        'r'
    ) as file:
        common_pos_patterns = [line.strip() for line in file.readlines()]
    
    results = unimodal_text_filtering(
        language_detection_model,
        tagging_model, 
        common_pos_patterns,
        tar_files, 
        log_folder,
        config)

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
