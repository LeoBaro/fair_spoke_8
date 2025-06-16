import ray
import pytest

from made.data_pipeline.actor_group import ActorGroup

def test_actor_group(config_path, log_folder, webdataset_output_folder, tar_files):
    actor_group = ActorGroup("UnimodalTextFilter", config_path, log_folder, webdataset_output_folder)
    assert ray.get_actor("UnimodalTextFilter_0") is not None
    assert ray.get_actor("UnimodalTextFilter_1") is not None

    actor_group.run(tar_files)
    tar_paths, uids_paths = actor_group.get_results()
    assert len(tar_paths) == 2
    assert len(uids_paths) == 2
    for tar_path in tar_paths:
        assert ".tar" in tar_path.name
    for uids_path in uids_paths:
        assert ".txt" in uids_path.name
        
    actor_group.kill_actors()
    with pytest.raises(Exception):
        ray.get_actor("UnimodalTextFilter_0")
        ray.get_actor("UnimodalTextFilter_1")
