import ray
import pytest

from made.data_pipeline.pipeline import ActorGroup, ActorGroupPipeline
from made.data_pipeline.steps.unimodal_text_filtering import UnimodalTextFilter


def test_actor_group(tar_files, config_path, log_folder):
    actor_group = ActorGroup("UnimodalTextFilter", 2, config_path)
    assert ray.get_actor("UnimodalTextFilter_0") is not None

    uids = None
    actor_group.execute_parallel(tar_files, log_folder, uids)
    results = actor_group.get_results()
    assert len(results) == 40

    uids = ['7e69a66689f697d961c18f085af23d8c', 'bfde488faeafb92deacaab312ff84e6b', '90f8bfaa173b7a1b74d0f12cd19e3808', '9fe96ee6087438d1f95ad26ba70ac38e']
    actor_group.execute_parallel(tar_files, log_folder, uids)
    results = actor_group.get_results()
    assert len(results) == 4

    actor_group.kill_actors()
    with pytest.raises(Exception):
        ray.get_actor("UnimodalTextFilter_0")
        ray.get_actor("UnimodalTextFilter_1")


def test_actor_group_pipeline_pretty_print(tar_files, config_path, log_folder):
    actor_group_pipeline = ActorGroupPipeline()
    
    actor_group_pipeline.add_pipeline_step("intersection")
    actor_group_pipeline.add_actor_group(0, "UnimodalTextFilter", 4, config_path)
    actor_group_pipeline.add_actor_group(0, "UnimodalVisionFilter", 3, config_path)
    
    actor_group_pipeline.add_pipeline_step("union")
    actor_group_pipeline.add_actor_group(1, "MultimodalFilter", 2, config_path)
    actor_group_pipeline.pretty_print()

    

def test_actor_group_pipeline(tar_files, config_path, log_folder):
    actor_group_pipeline = ActorGroupPipeline()
    actor_group_pipeline.add_pipeline_step("intersection")
    actor_group_pipeline.add_actor_group(0, "UnimodalTextFilter", 2, config_path)
    uids = actor_group_pipeline.execute(tar_files, log_folder)
    assert len(uids) == 40

    actor_group_pipeline.add_actor_group(0, "UnimodalVisionFilter", 1, config_path)
    uids = actor_group_pipeline.execute(tar_files, log_folder)
    assert len(uids) == 30


    actor_group_pipeline.add_pipeline_step("union")
    actor_group_pipeline.add_actor_group(1, "MultimodalFilter", 1, config_path)
    uids = actor_group_pipeline.execute(tar_files, log_folder)
    assert len(uids) == 3

    