import argparse
import logging
from time import time
from pathlib import Path
import traceback

from made.config import Config
from made.bin.cli import cli
from made.bin.made import make_pipeline
from made.data_pipeline.utils import connect_or_start_ray, collect_tar_files, save_uids, cleanup
from made.data_pipeline.pipeline import ActorGroupPipeline
from utils import create_config, create_output_folder, create_results_and_log_folders, create_result_file

import os
os.environ["RAY_DEDUP_LOGS"] = "0"

def make_single_step_pipeline(config_path: str | Path, filtering_step_name: str, config: dict):
    if filtering_step_name == "UnimodalTextFilter":
        num_workers = config.unimodal_text.num_workers
    elif filtering_step_name == "UnimodalVisionFilter":
        num_workers = config.unimodal_vision.num_workers
    elif filtering_step_name == "MultimodalFilter":
        num_workers = config.multimodal.num_workers

    actor_group_pipeline = ActorGroupPipeline()
    actor_group_pipeline.add_pipeline_step("intersection")
    actor_group_pipeline.add_actor_group(0, filtering_step_name, num_workers, config_path)
    return actor_group_pipeline

def execution_loop(filtering_step_name, unimodal_text_num_workers: list[int], unimodal_vision_num_workers: list[int], multimodal_num_workers: list[int], text_batch_sizes: list[int], vision_batch_sizes: list[int], multimodal_batch_sizes: list[int], num_executions: int, shards_path: str, output_folder: Path, result_file: Path, enable_metrics: bool, save_filtered_uids: bool):
    for unimodal_text_num_workers, unimodal_vision_num_workers, multimodal_num_workers in zip(unimodal_text_num_workers, unimodal_vision_num_workers, multimodal_num_workers):
        for text_batch_size, vision_batch_size, multimodal_batch_size in zip(text_batch_sizes, vision_batch_sizes, multimodal_batch_sizes):
            config = {
                "unimodal_text_num_workers": unimodal_text_num_workers,
                "unimodal_vision_num_workers": unimodal_vision_num_workers,
                "multimodal_num_workers": multimodal_num_workers,
                "text_batch_size": text_batch_size,
                "vision_batch_size": vision_batch_size,
                "multimodal_batch_size": multimodal_batch_size,
                "enable_metrics": enable_metrics,
                "save_filtered_uids": save_filtered_uids
            }
            for iteration_index in range(num_executions):
                current_output_folder, current_log_folder = create_results_and_log_folders(output_folder, suffix=f"_nw{unimodal_text_num_workers}_{unimodal_vision_num_workers}_{multimodal_num_workers}_bs{text_batch_size}_{vision_batch_size}_{multimodal_batch_size}_i{iteration_index}")
                try:
                    print("Running pipeline..")
                    took = run_pipeline(filtering_step_name, shards_path, current_output_folder, config)
                    with open(result_file, "a", encoding="utf-8") as f:
                        f.write(f"{unimodal_text_num_workers},{unimodal_vision_num_workers},{multimodal_num_workers},{text_batch_size},{vision_batch_size},{multimodal_batch_size},{iteration_index},{took}\n")
                    cleanup()
                except Exception as e:
                    print(f"Error running pipeline: {e}")
                    traceback.print_exc()
                    cleanup()

def run_pipeline(filtering_step_name: str, shards_path: str, output_folder: Path, config: dict):
    config_path = create_config(config, output_folder)
    config = Config(config_path)

    connect_or_start_ray(None, Config().infrastructure.logging_level)

    logger = logging.getLogger("ray")
    logger.info("Starting pipeline")
    logger.info("Unimodal text num workers: %s", config.unimodal_text.num_workers)
    logger.info("Unimodal vision num workers: %s", config.unimodal_vision.num_workers)
    logger.info("Multimodal num workers: %s", config.multimodal.num_workers)

    if filtering_step_name == "MadePipeline":
        made_pipeline = make_pipeline(config_path)
    else:
        made_pipeline = make_single_step_pipeline(config_path, filtering_step_name, config)

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
        choices=["UnimodalVisionFilter", "UnimodalTextFilter", "MultimodalFilter", "MadePipeline"]
    )
    parser.add_argument(
        "-n",
        "--num_executions",
        type=int,
        required=False,
        default=2
    )
    parser.add_argument(
        "-utw",
        "--unimodal_text_num_workers",
        nargs="+",
        type=int,
        required=False,
        default=[1,2,4]
    )
    parser.add_argument(
        "-uvw",
        "--unimodal_vision_num_workers",
        nargs="+",
        type=int,
        required=False,
        default=[1,2,4]
    )
    parser.add_argument(
        "-mw",
        "--multimodal_num_workers",
        nargs="+",
        type=int,
        required=False,
        default=[1,2,4]
    )
    parser.add_argument(
        "-tbs",
        "--text_batch_sizes",
        nargs="+",
        type=int,
        required=False,
        default=[2000]
    )
    parser.add_argument(
        "-vbs",
        "--vision_batch_sizes",
        nargs="+",
        type=int,
        required=False,
        default=[500]
    )
    parser.add_argument(
        "-mbs",
        "--multimodal_batch_sizes", 
        nargs="+",
        type=int,
        required=False,
        default=[300]
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
        type=bool,
        required=False,
        default=False,
        help="Enable metrics and save filtered uids to perform quality tests"
    )
    parser.add_argument(
        "--save_filtered_uids",
        type=bool,
        required=False,
        default=False,
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
        args.unimodal_text_num_workers,
        args.unimodal_vision_num_workers,
        args.multimodal_num_workers,
        args.text_batch_sizes,
        args.vision_batch_sizes,
        args.multimodal_batch_sizes,
        args.num_executions,
        args.shards_path,
        output_folder,
        result_file,
        args.enable_metrics,
        args.save_filtered_uids
    )

    # test on batch size scalability for unimodal text filter
    # python run_benchmark.py -f UnimodalTextFilter -n 2 -utw 1 -uvw 1 -mw 1 -tbs 100 1000 10000 -vbs 100 1000 10000 -mbs 100 1000 10000 

    # python run_benchmark.py -f MadePipeline -n 2 -utw 1 2 -uvw 1 2 -mw 1 2 -tbs 2000 -vbs 500 -mbs 300 -s /home/leobaro/workspace/labs/fair_spoke_8/data_quality_pipeline/benchmark/data --enable_metrics --save_filtered_uids

if __name__ == "__main__":
    args = cli()
    main(args)