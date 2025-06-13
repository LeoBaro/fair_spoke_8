import pytest
import ray
import torch

from made.data_pipeline.steps.specificity_filtering import (
    specificity_filtering,
    SpecificityFilter
)
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import FilteringResult
from made.data_pipeline.steps.specificity_filtering import model_init
from made.paths import MADE_PATH

# ------------------------
# Test without Ray
# ------------------------

def test_specificity_filtering(tar_files, log_folder, webdataset_output_folder, config):

    ref = torch.load(str(MADE_PATH / config.specificity.reference_path))
    img_ref = ref["img"].to("cuda")
    txt_ref = ref["txt"].to("cuda")

    # Load model
    model, trs = model_init(pretrained=str(MADE_PATH / config.specificity.model_path))
    model = model.to("cuda").eval()

    metrics_store = MetricsStore(log_folder)
    filtering_result = FilteringResult(webdataset_output_folder, config.infrastructure.dump_tar_every_n_samples)

    produced_tar_files, produced_uids_files = specificity_filtering(
        tar_files=tar_files,
        model=model,
        trs=trs,
        img_ref=img_ref,
        txt_ref=txt_ref,
        config=config,
        metrics_store=metrics_store,
        filtering_result=filtering_result
    )

    assert len(produced_tar_files) == 1
    assert len(produced_uids_files) == 1
    assert produced_tar_files[0].exists()
    assert produced_uids_files[0].exists()


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
