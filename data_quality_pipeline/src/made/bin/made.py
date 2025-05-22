import atexit
import ray
import logging
import argparse
from time import time
from pathlib import Path

from made.config import Config
from made.bin.cli import cli
from made.data_pipeline.utils import connect_or_start_ray, collect_tar_files, save_uids, shutdown_ray, cleanup
from made.data_pipeline.pipeline import ActorGroupPipeline


def make_pipeline(config_path: str | Path):

    actor_group_pipeline = ActorGroupPipeline()

    actor_group_pipeline.add_pipeline_step("intersection")
    actor_group_pipeline.add_actor_group(0, "UnimodalTextFilter", 2, config_path)
    actor_group_pipeline.add_actor_group(0, "UnimodalVisionFilter", 1, config_path)


    actor_group_pipeline.add_pipeline_step("union")
    actor_group_pipeline.add_actor_group(1, "MultimodalFilter", 1, config_path)

    return actor_group_pipeline

def main(args):
    config = Config(args.config_path)

    connect_or_start_ray(args.ray_address, Config().infrastructure.logging_level)

    #atexit.register(save_aggregated_metrics)

    logger = logging.getLogger("ray")
    logger.info("Starting pipeline")
    logger.info("Num workers: %s", config.infrastructure.num_workers)
    
    made_pipeline = make_pipeline(args.config_path)

    s = time()
    good_uids = made_pipeline.execute(
        collect_tar_files(args.shards_path), 
        args.log_folder
    )
    took = time() - s
    logger.info(f"Pipeline completed. Took {took:0.2f} seconds")


    logger.info("Saving uids")
    save_uids(good_uids, args.output_folder)

    cleanup()

    return took

    # TODO: After creating a subset, you may invoke the resharder to build the subset shards 
    # From: https://github.com/mlfoundations/datacomp
    # in $output_dir like so: 
    # python resharder.py -i $download_dir -o $output_dir -s $subset_file

if __name__ == "__main__":
    args = cli()
    main(args)