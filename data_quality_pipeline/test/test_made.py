import pytest
from argparse import Namespace

from made.bin.made import main


def test_ray_made_pipeline(ray_init, data_path, output_folder, log_folder, config_path):

    # if not ray_flag:
    #     pytest.skip("Skipping Ray test because --ray flag was not provided.") 
    
    args = Namespace(   
        shards_path=data_path,
        ray_address=None,
        log_folder=log_folder,
        output_folder=output_folder,
        config_path=str(config_path)
    )
    
    tar_paths, uids_paths, output_filename = main(args)

    assert output_filename.exists()
    assert ".npy" in output_filename.name

