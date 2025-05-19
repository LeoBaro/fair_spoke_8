import pytest
import logging
import ray
from made.paths import MADE_PATH

from made.data_pipeline.steps.semdedup_filtering import (
    remove_image_duplicates,
    SemDeDupFilter
)
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def test_unimodal_text_filtering(tar_files, log_folder, config):

    results = remove_image_duplicates(
        tar_files,
        log_folder,
        config
    )