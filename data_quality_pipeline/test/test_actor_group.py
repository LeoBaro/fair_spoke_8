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
