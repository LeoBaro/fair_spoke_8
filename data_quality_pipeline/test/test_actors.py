import ray
import pytest

from made.data_pipeline.actors.unimodal_text_filtering import UnimodalTextFilter
from made.data_pipeline.actors.unimodal_vision_filtering import UnimodalVisionFilter
from made.data_pipeline.actors.multimodal_alignment_filtering import MultimodalAlignmentFilter
from made.data_pipeline.actors.multimodal_specificity_filtering import MultimodalSpecificityFilter
from made.data_pipeline.actors.semantic_dedup_filtering import SemanticDedupFilter

def test_unimodal_text_filter_actor(ray_init, config_path, log_folder, webdataset_output_folder, tar_files):
    
    unimodalTextFilter = UnimodalTextFilter.options(
        name="UnimodalTextFilter",
        num_gpus=1,
        max_concurrency=1
    ).remote(
        config_path, 
        log_folder, 
        webdataset_output_folder
    )

    results = ray.get(
        [
            unimodalTextFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files, produced_uids_files = results[0]
    assert len(produced_tar_files) == 1
    assert len(produced_uids_files) == 1
    assert produced_tar_files[0].exists()
    assert produced_uids_files[0].exists()

def test_unimodal_vision_filter_actor(ray_init, config_path, log_folder, webdataset_output_folder, tar_files):
    
    unimodalVisionFilter = UnimodalVisionFilter.options(
        name="UnimodalVisionFilter",
        num_gpus=1,
        max_concurrency=1
    ).remote(config_path, log_folder, webdataset_output_folder)

    # single worker test 
    results = ray.get(
        [
            unimodalVisionFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files, produced_uids_files = results[0]
    assert len(produced_tar_files) == 1
    assert len(produced_uids_files) == 1
    assert produced_tar_files[0].exists()
    assert produced_uids_files[0].exists()


def test_ray_multimodal_filtering(ray_init, tar_files, log_folder, webdataset_output_folder, config_path):
    
    multimodalAlignmentFilter = MultimodalAlignmentFilter.options(
        name="MultimodalAlignmentFilter",
        num_gpus=1,
        max_concurrency=1
    ).remote(config_path, log_folder, webdataset_output_folder)

    results = ray.get(
        [
            multimodalAlignmentFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files, produced_uids_files = results[0]
    assert len(produced_tar_files) == 1
    assert len(produced_uids_files) == 1
    assert produced_tar_files[0].exists()
    assert produced_uids_files[0].exists()

def test_ray_multimodal_specificity_filtering(ray_init, tar_files, log_folder, webdataset_output_folder, config_path):
    
    multimodalSpecificityFilter = MultimodalSpecificityFilter.options(
        name="MultimodalSpecificityFilter",
        num_gpus=1,
        max_concurrency=1
    ).remote(config_path, log_folder, webdataset_output_folder)

    results = ray.get(
        [
            multimodalSpecificityFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files, produced_uids_files = results[0]
    assert len(produced_tar_files) == 1
    assert len(produced_uids_files) == 1
    assert produced_tar_files[0].exists()
    assert produced_uids_files[0].exists()

def test_ray_semantic_dedup_filtering(
        ray_init, 
        tar_files,
        log_folder, 
        webdataset_output_folder, 
        config_path
    ):
    
    semanticDedupFilter = SemanticDedupFilter.options(
        name="SemanticDedupFilter",
        num_gpus=1,
        max_concurrency=1
    ).remote(config_path, log_folder, webdataset_output_folder)
    
    results = ray.get(
        [
            semanticDedupFilter.execute.remote(tar_files)
        ]
    )
    produced_tar_files, produced_uids_files = results[0]
    assert len(produced_tar_files) == 1
    assert len(produced_uids_files) == 1
    assert produced_tar_files[0].exists()
    assert produced_uids_files[0].exists()