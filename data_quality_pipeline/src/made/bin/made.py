import atexit
import ray
import logging
from time import time

from made.config import Config
from made.data_pipeline.utils import collect_tar_files, save_uids, shutdown_ray
from made.data_pipeline.pipeline import run_pipeline

import argparse
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_path", required=True, nargs="+")
    parser.add_argument("--ray-address", type=str, required=True)
    parser.add_argument("--log-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=False, default=None)
    return parser.parse_args()

def connect_or_start_ray(args):
    if args.ray_address:
        print("Connecting to Ray at", args.ray_address)
        ray.init(
            address=args.ray_address,
            logging_level=getattr(logging, Config().infrastructure.logging_level),
            log_to_driver=True
        )
    else:
        print("Starting Ray locally")
        ray.init(
            logging_level=getattr(logging, Config().infrastructure.logging_level),
            log_to_driver=True
        )

def main(args):
    config = Config(args.config_path)
    connect_or_start_ray(args)

    #atexit.register(save_aggregated_metrics)

    logger = logging.getLogger("ray")
    logger.info("Starting pipeline")
    
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

    shutdown_ray()
    
    return took

    # TODO: After creating a subset, you may invoke the resharder to build the subset shards 
    # From: https://github.com/mlfoundations/datacomp
    # in $output_dir like so: 
    # python resharder.py -i $download_dir -o $output_dir -s $subset_file

if __name__ == "__main__":
    args = cli()
    main(args)