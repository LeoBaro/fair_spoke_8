import logging
from time import time

from made.config import Config
from made.bin.cli import cli
from made.data_pipeline.utils import connect_or_start_ray, collect_tar_files, save_uids, cleanup
from made.data_pipeline.actor_group import ActorGroup

def main(args):
    config = Config(args.config_path)

    connect_or_start_ray(args.ray_address, Config().infrastructure.logging_level, Config().infrastructure.log_to_driver, args.log_folder)

    #atexit.register(save_aggregated_metrics)

    logger = logging.getLogger("ray")

    logger.info("Configuration:\n %s", config)
    
    actor_group = ActorGroup("UnimodalTextFilter", 2, args.config_path, args.log_folder, args.output_folder)

    s = time()
    actor_group.run(
        collect_tar_files(args.shards_path),
    )
    tar_paths, uids_paths = actor_group.get_results()
    took = time() - s
    logger.info("Pipeline completed. Took %0.2f seconds", took)


    # logger.info("Saving uids to %s", args.output_folder)
    # good_uids = []
    # for uids_path in uids_paths:
    #     with open(uids_path, "r", encoding="utf-8") as f:
    #         good_uids.extend(f.readlines())
    # output_filename = save_uids(good_uids, args.output_folder)

    cleanup()

    return tar_paths, uids_paths

    # TODO: After creating a subset, you may invoke the resharder to build the subset shards 
    # From: https://github.com/mlfoundations/datacomp
    # in $output_dir like so: 
    # python resharder.py -i $download_dir -o $output_dir -s $subset_file

if __name__ == "__main__":
    args = cli()
    main(args)