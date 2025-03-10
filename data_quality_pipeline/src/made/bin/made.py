import atexit
import ray
import logging
import argparse
from time import time

from made.config import Config
from made.data_pipeline.utils import collect_tar_files, save_uids, shutdown_ray
from made.data_pipeline.pipeline import run_pipeline
from made.data_pipeline.common import Singleton


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_path", required=True, nargs="+")
    parser.add_argument("--ray-address", type=str, required=True)
    parser.add_argument("--log-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=False, default=None)
    return parser.parse_args()

def connect_or_start_ray(ray_address, logging_level):
    if ray_address:
        print("Connecting to Ray at", ray_address)
        ray.init(
            address=ray_address,
            logging_level=getattr(logging, logging_level),
            log_to_driver=True
        )
    else:
        print("Starting Ray locally")
        ray.init(
            logging_level=getattr(logging, logging_level),
            log_to_driver=True
        )

def cleanup():
    shutdown_ray()
    Singleton.destroy_instance(Config)

def main(args):
    config = Config(args.config_path)

    connect_or_start_ray(args.ray_address, Config().infrastructure.logging_level)

    #atexit.register(save_aggregated_metrics)

    logger = logging.getLogger("ray")
    logger.info("Starting pipeline")
    logger.info("Num workers: %s", config.infrastructure.num_workers)
    
    s = time()
    ok_uids = run_pipeline(
        collect_tar_files(args.shards_path), 
        config.infrastructure.num_workers,
        args.log_folder,
        args.config_path
    )
    took = time() - s
    logger.info(f"Pipeline completed. Took {took:0.2f} seconds")


    if config.infrastructure.save_npy:
        logger.info("Saving uids")
        save_uids(ok_uids, args.output_folder)

    cleanup()

    return took

    # TODO: After creating a subset, you may invoke the resharder to build the subset shards 
    # From: https://github.com/mlfoundations/datacomp
    # in $output_dir like so: 
    # python resharder.py -i $download_dir -o $output_dir -s $subset_file

if __name__ == "__main__":
    args = cli()
    main(args)