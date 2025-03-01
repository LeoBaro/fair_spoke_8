import pytest
from argparse import Namespace

from made.bin.made import main
from made.config import Config


def test_ray_made_pipeline(configure_for_test, data_path, output_folder, log_folder, config_path):
    args = Namespace(   
        shards_path=data_path,
        ray_address=None,
        log_folder=log_folder,
        output_folder=output_folder,
        config_path=str(config_path)
    )
    
    main(args)
    
    assert output_folder.exists()
    assert output_folder.joinpath("ok_uids.npy").exists()
    assert log_folder.exists()
    # Add node id 
    # assert log_folder.joinpath("metrics_summary.json").exists()
    # assert log_folder.joinpath("metrics_details.json").exists()
    