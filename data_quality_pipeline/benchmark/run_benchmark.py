import argparse
import logging
from time import time
from pathlib import Path
import traceback

from made.config import Config
from made.bin.cli import cli
from made.data_pipeline.utils import connect_or_start_ray, collect_tar_files, save_uids, cleanup
from made.data_pipeline.pipeline import ActorGroupPipeline
from utils import create_config, create_output_folder, create_results_and_log_folders, create_result_file

import os
os.environ["RAY_DEDUP_LOGS"] = "0"

def make_single_step_pipeline(config_path: str | Path, filtering_step_name: str, num_workers: int):
    actor_group_pipeline = ActorGroupPipeline()
    actor_group_pipeline.add_pipeline_step("intersection")
    actor_group_pipeline.add_actor_group(0, filtering_step_name, num_workers, config_path)
    return actor_group_pipeline

def execution_loop(filtering_step_name, workers: list[int], batch_sizes: list[int], num_executions: int, shards_path: str, output_folder: Path, result_file: Path):
    for num_workers in workers:
        for batch_size in batch_sizes:
            for iteration_index in range(num_executions):
                current_output_folder, current_log_folder = create_results_and_log_folders(output_folder, suffix=f"_nw{num_workers}_bs{batch_size}_i{iteration_index}")
                try:
                    print("Running pipeline..")
                    took = run_pipeline(filtering_step_name, num_workers, batch_size, shards_path, current_output_folder)
                    with open(result_file, "a", encoding="utf-8") as f:
                        f.write(f"{num_workers},{batch_size},{iteration_index},{took}\n")
                    cleanup()
                except Exception as e:
                    print(f"Error running pipeline: {e}")
                    traceback.print_exc()
                    cleanup()

def run_pipeline(filtering_step_name: str, num_workers: int, batch_size: int, shards_path: str, output_folder: Path):
    config_path = create_config(num_workers, batch_size)
    config = Config(config_path)

    connect_or_start_ray(None, Config().infrastructure.logging_level)

    logger = logging.getLogger("ray")
    logger.info("Starting pipeline")
    logger.info("Num workers: %s", config.infrastructure.num_workers)
    
    made_pipeline = make_single_step_pipeline(config_path, filtering_step_name, num_workers)

    s = time()
    ok_uids = made_pipeline.execute(
        collect_tar_files(shards_path, recursive=False),
        output_folder
    )
    took = time() - s
    logger.info(f"Pipeline completed. Took {took:0.2f} seconds")


    logger.info("Saving uids")
    # breakpoint()
    save_uids(ok_uids, output_folder)

    return took




def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filtering_step_name", 
        type=str, 
        required=True,
        choices=["UnimodalVisionFilter", "UnimodalTextFilter"]
    )
    return parser.parse_args()



def main(args):

    num_executions = 1
    num_workers_list = [1]
    batch_sizes = [300]
    # shards_path = "/home/leobaro/Downloads/datasets/web/datacomp/_completed_downloads"
    # shards_path = "/home/leobaro/workspace/labs/fair_spoke_8/data_quality_pipeline/test"
    shards_path = "/home/leobaro/workspace/labs/fair_spoke_8/data_quality_pipeline/benchmark/data"
    output_folder = create_output_folder()
    result_file = create_result_file(output_folder)

    cleanup()

    execution_loop(
        args.filtering_step_name,
        num_workers_list,
        batch_sizes,
        num_executions,
        shards_path,
        output_folder, 
        result_file
    )


if __name__ == "__main__":
    args = cli()
    main(args)