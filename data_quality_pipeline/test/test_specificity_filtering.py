import pytest
import ray
import torch
from pathlib import Path

from data_quality_pipeline.src.made.config import Config
from data_quality_pipeline.src.made.data_pipeline.steps.specificity_filtering import (
    specificity_filtering,
    SpecificityFilter
)

# ------------------------
# Test without Ray
# ------------------------

def test_specificity_filtering(tar_files, log_folder, config):
    ref_path = "/davinci-1/work/fdimatteo/hype_weights/reference.pt"
    ref = torch.load(ref_path)
    img_ref, txt_ref = ref["img"], ref["txt"]

    results = specificity_filtering(
        tar_files=tar_files,
        log_folder=log_folder,
        config=config,
        img_ref=img_ref,
        txt_ref=txt_ref
    )

    assert isinstance(results, list)
    # Optional: Add expected result count or value-type check
    # assert len(results) == expected_count


# ------------------------
# Test with Ray
# ------------------------

def test_ray_specificity_filtering(ray_init, ray_flag, tar_files, log_folder, config_path):
    if not ray_flag:
        pytest.skip("Skipping Ray test because --ray flag was not provided.")

    specificity_filter = SpecificityFilter.remote(config_path)

    results = ray.get([
        specificity_filter.execute.remote(tar_files, log_folder)
    ])

    assert isinstance(results[0], list)
    # Optional: Validate result content
    # assert len(results[0]) == expected_count
