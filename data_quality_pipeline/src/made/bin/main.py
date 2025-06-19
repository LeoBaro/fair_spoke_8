import logging
from time import time
from pathlib import Path
import argparse

from made.config import Config
from made.data_pipeline.utils import connect_or_start_ray, collect_tar_files, cleanup, remove_and_recreate_dirs
from made.data_pipeline.actor_group import ActorGroup


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering-step-name", type=str, required=True, choices=["UnimodalTextFilter", "UnimodalVisionFilter", "MultimodalAlignmentFilter", "MultimodalSpecificityFilter"])
    parser.add_argument("--shards-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--log-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--ray-address", type=str, required=False, default=None)
    return parser.parse_args()

def main(args):
    config = Config(args.config_path)

    remove_and_recreate_dirs([args.output_folder, args.log_folder])

    connect_or_start_ray(
        args.ray_address, 
        Config().infrastructure.logging_level, 
        Config().infrastructure.log_to_driver, 
        Path(args.log_folder),
        int(float(Config().infrastructure.ray_object_store_memory))
    )

    logger = logging.getLogger("ray")

    logger.info("Configuration:\n %s", config)
    
    actor_group = ActorGroup(
        args.filtering_step_name,
        args.config_path,
        args.log_folder,
        args.output_folder
    )

    s = time()
    actor_group.run(
        collect_tar_files(args.shards_path, recursive=True),
    )
    tar_paths, uids_paths = actor_group.get_results()
    took = time() - s
    logger.info("Pipeline completed. Took %0.2f seconds. Generated %d shards.", took, len(tar_paths))

    cleanup()

    return tar_paths, uids_paths

    # TODO: Save uids to a file
    # logger.info("Saving uids to %s", args.output_folder)
    # good_uids = []
    # for uids_path in uids_paths:
    #     with open(uids_path, "r", encoding="utf-8") as f:
    #         good_uids.extend(f.readlines())
    # output_filename = save_uids(good_uids, args.output_folder)


    # TODO: After creating a subset, you may invoke the resharder to build the subset shards 
    # From: https://github.com/mlfoundations/datacomp
    # in $output_dir like so: 
    # python resharder.py -i $download_dir -o $output_dir -s $subset_file

if __name__ == "__main__":
    args = cli()
    main(args)