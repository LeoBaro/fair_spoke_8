import logging
import argparse
import traceback
from time import time
from pathlib import Path

from made.config import Config
from made.data_pipeline.utils import connect_or_start_ray, collect_tar_files, cleanup
from made.data_pipeline.actor_group import ActorGroup
from utils import create_output_folder, create_results_and_log_folders, create_result_file

import os
os.environ["RAY_DEDUP_LOGS"] = "0"

def execution_loop(
        filtering_step_name, 
        num_workers: list[int], 
        batch_size: int, 
        num_executions: int, 
        shards_path: str, 
        output_folder: Path, 
        result_file: Path, 
        enable_metrics: bool, 
        save_bad_uids: bool
    ):
    count = 0
    for nw in num_workers:
            config = {
                "num_workers": nw,
                "batch_size": batch_size,
                "enable_metrics": enable_metrics,
                "save_bad_uids": save_bad_uids
            }
            for iteration_index in range(num_executions):
                current_output_folder, current_log_folder = create_results_and_log_folders(output_folder, suffix=f"_nw{nw}_bs{batch_size}_i{iteration_index}")
                try:
                    print("Running pipeline..")
                    took = run_actor_group(count, filtering_step_name, shards_path, current_output_folder, current_log_folder, config)
                    with open(result_file, "a", encoding="utf-8") as f:
                        f.write(f"{nw},{batch_size},{iteration_index},{took}\n")
                    cleanup()
                except Exception as e:
                    print(f"Error running pipeline: {e}")
                    traceback.print_exc()
                    cleanup()
                count += 1

def run_actor_group(index: int, filtering_step_name: str, shards_path: str, output_folder: Path, log_folder: Path, override_config: dict):
    config_path = Config.create_config(override_config, output_folder / f"config_{index}.yaml")
    config = Config(config_path)

    connect_or_start_ray(None, config.infrastructure.logging_level, config.infrastructure.log_to_driver, log_folder)

    logger = logging.getLogger("ray")

    actor_group = ActorGroup(filtering_step_name, config_path, log_folder, output_folder)

    s = time()
    actor_group.run(collect_tar_files(shards_path, recursive=False))
    results = actor_group.get_results() 
    took = time() - s
    logger.info(f"Pipeline completed. Took {took:0.2f} seconds")

    return took


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filtering_step_name", 
        type=str, 
        required=True,
        choices=["UnimodalVisionFilter", "UnimodalTextFilter", "MultimodalFilter"]
    )
    parser.add_argument(
        "-n",
        "--num_executions",
        type=int,
        required=False,
        default=2
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        nargs="+",
        type=int,
        required=False,
        default=[1,2,4]
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=500
    )
    parser.add_argument(
        "-s",
        "--shards_path",
        type=str,
        required=False,
        default="/home/leobaro/workspace/labs/fair_spoke_8/data_quality_pipeline/benchmark/data"
    )
    parser.add_argument(
        "--enable_metrics",
        action="store_true",
        required=False,
        help="Enable metrics and save filtered uids to perform quality tests"
    )
    parser.add_argument(
        "--save_bad_uids",
        action="store_true",
        required=False,
        help="Save filtered uids to perform quality tests"
    )
    
    
    return parser.parse_args()



def main(args):

    # python scalability_visualization.py -f out_*_filtername/benchmark_results.csv -t "Filter name"
    # python samples_visualization.py -r out_*_filtername 
    cleanup()
    output_folder = create_output_folder(args.filtering_step_name)
    result_file = create_result_file(output_folder)
  
    execution_loop(
        args.filtering_step_name,
        args.num_workers,
        args.batch_size,
        args.num_executions,
        args.shards_path,
        output_folder,
        result_file,
        args.enable_metrics,
        args.save_bad_uids
    )

    # test on batch size scalability
    # python run_benchmark.py -f UnimodalTextFilter -w 2 -bs 100 1000 10000 


if __name__ == "__main__":
    args = cli()
    main(args)