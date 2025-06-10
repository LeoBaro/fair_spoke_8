import ray
import pytest

from made.data_pipeline.actor_group import ActorGroup

def test_actor_group(config_path, log_folder, webdataset_output_folder, tar_files):
    actor_group = ActorGroup("UnimodalTextFilter", 2, config_path, log_folder, webdataset_output_folder)
    assert ray.get_actor("UnimodalTextFilter_0") is not None
    assert ray.get_actor("UnimodalTextFilter_1") is not None

    actor_group.run(tar_files)
    results = actor_group.get_results()
    assert len(results) == 2
    
    actor_group.kill_actors()
    with pytest.raises(Exception):
        ray.get_actor("UnimodalTextFilter_0")
        ray.get_actor("UnimodalTextFilter_1")


# def test_actor_group_pipeline_pretty_print(tar_files, config_path, log_folder):
#     actor_group_pipeline = ActorGroupPipeline()
    
#     actor_group_pipeline.add_pipeline_step("intersection")
#     actor_group_pipeline.add_actor_group(0, "UnimodalTextFilter", 4, config_path)
#     actor_group_pipeline.add_actor_group(0, "UnimodalVisionFilter", 3, config_path)
    
#     actor_group_pipeline.add_pipeline_step("union")
#     actor_group_pipeline.add_actor_group(1, "MultimodalFilter", 2, config_path)
#     actor_group_pipeline.pretty_print()

    

# def test_actor_group_pipeline(tar_files, config_path, log_folder):
#     actor_group_pipeline = ActorGroupPipeline()
#     actor_group_pipeline.add_pipeline_step("intersection")
#     actor_group_pipeline.add_actor_group(0, "UnimodalTextFilter", 2, config_path)
#     uids = actor_group_pipeline.execute(tar_files, log_folder)
#     assert len(uids) == 40

#     actor_group_pipeline.add_actor_group(0, "UnimodalVisionFilter", 1, config_path)
#     uids = actor_group_pipeline.execute(tar_files, log_folder)
#     assert len(uids) == 25


#     actor_group_pipeline.add_pipeline_step("union")
#     actor_group_pipeline.add_actor_group(1, "MultimodalFilter", 1, config_path)
#     uids = actor_group_pipeline.execute(tar_files, log_folder)
#     assert len(uids) == 18

    