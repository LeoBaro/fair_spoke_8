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
from made.data_pipeline.steps.base import FilteringResult

def test_unimodal_text_filtering(tar_files, log_folder, webdataset_output_folder, config):
    language_detection_model = fasttext.load_model(
        str(MADE_PATH / config.unimodal_text.lang_detection_model_path)
        )
    spacy.require_gpu()
    tagging_model = spacy.load(
        str(config.unimodal_text.tagging_model_name)
        )
    with open(
        str(MADE_PATH / config.unimodal_text.good_captions_pos_distribution_path),
        'r',
        encoding='utf-8'
    ) as file:
        common_pos_patterns = [line.strip() for line in file.readlines()]
    
    metrics_store = MetricsStore(log_folder)
    filtering_result = FilteringResult(webdataset_output_folder, config.infrastructure.dump_tar_every_n_samples)

    produced_tar_files = unimodal_text_filtering(
        tar_files,
        language_detection_model,
        tagging_model, 
        common_pos_patterns,
        config,
        metrics_store,
        filtering_result
    )
    assert len(produced_tar_files) == 1

def test_ray_unimodal_text_filtering(ray_init, ray_flag, tar_files, log_folder, webdataset_output_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")    
    
    unimodalTextFilter = UnimodalTextFilter.remote(config_path, log_folder, webdataset_output_folder)

    results = ray.get(
        [
            unimodalTextFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files = results[0]
    assert len(produced_tar_files) == 1
