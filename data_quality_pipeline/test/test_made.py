import pytest
from argparse import Namespace

from made.bin.main import main


def test_ray_made_pipeline(data_path, output_folder, log_folder, get_config_path):

    args = Namespace(   
        filtering_step_name="UnimodalTextFilter",
        shards_path=data_path,
        config_path=get_config_path(),
        log_folder=log_folder,
        output_folder=output_folder,
        ray_address=None
    )

    tar_paths, uids_paths = main(args)

    for tar_path, uids_path in zip(tar_paths, uids_paths):
        assert tar_path.exists()
        assert uids_path.exists()

