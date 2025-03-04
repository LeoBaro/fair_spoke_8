import atexit
import ray
import logging
from made.bin.cli import cli
from made.config import Config
from made.data_pipeline.utils import collect_tar_files, save_uids, shutdown_ray
from made.data_pipeline.pipeline import run_pipeline

def connect_or_start_ray(args):
    if args.ray_address:
        print("Connecting to Ray at", args.ray_address)
        ray.init(
            address=args.ray_address,
            logging_level=getattr(logging, Config().infrastructure.logging_level)
        )
    else:
        print("Starting Ray locally")
        ray.init(
            logging_level=getattr(logging, Config().infrastructure.logging_level)
        )

def main(args):
    config = Config(args.config_path)
    connect_or_start_ray(args)
    atexit.register(shutdown_ray)
    #atexit.register(save_aggregated_metrics)

    logger = logging.getLogger("main")
    logger.info("Starting pipeline")

    ok_uids = run_pipeline(
        collect_tar_files(args.shards_path), 
        config.infrastructure.num_workers,
        config.infrastructure.log_folder
    )

    logger.info("Saving uids")
    save_uids(ok_uids, config.infrastructure.output_folder)

    logger.info("Pipeline completed")

    # TODO: After creating a subset, you may invoke the resharder to build the subset shards 
    # From: https://github.com/mlfoundations/datacomp
    # in $output_dir like so: 
    # python resharder.py -i $download_dir -o $output_dir -s $subset_file

if __name__ == "__main__":
    args = cli()
    main(args)