import os
import json
import numpy as np
from pathlib import Path
from argparse import Namespace
from datetime import datetime

os.environ["RAY_DEDUP_LOGS"] = "0"

def create_config(num_workers, batch_size):
    config_raw = f"""
infrastructure:
    num_workers: {num_workers} 
    enable_metrics: false
    logging_level: WARNING
    save_npy: false
    apply_filters: false
    
unimodal:
    batch_size: {batch_size} 

    caption_min_words: 2
    caption_min_chars: 5

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    tagging_model_name: en_core_web_trf
    good_captions_pos_distribution_path: models/common_pos_patterns.txt

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 1.8

multimodal:
    batch_size: 32
    clip_model: openai/clip-vit-base-patch32
    clip_score_threshold: 0.3
"""
    
    config_file_path = "/tmp/single_node_config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write(config_raw)
    return config_file_path

def create_output_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(__file__).parent / f"out_{timestamp}"
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder

def create_results_and_log_folders(output_folder: Path, suffix: str = ""):
    results_folder = output_folder / "results"
    log_folder = output_folder / "logs"
    results_folder.mkdir(exist_ok=True, parents=True)
    log_folder.mkdir(exist_ok=True, parents=True)
    return results_folder, log_folder

if __name__=='__main__':

    # batch size = 30000 is greater than the total number of samples split (25K)

    num_executions = 2
    num_workers_list = [16, 32] 
    shards_path = "/davinci-1/work/lbaroncelli/datacomp_32_tar_files"
    shards_path = "/home/leobaro/workspace/labs/fair_spoke_8/data_quality_pipeline/benchmark/data"

    from made.bin.made import main

    output_folder = create_output_folder()
    results_file = output_folder / "benchmark_results.json"


    results = {}
    
    num_workers_baseline = num_workers_list[0] 
    batch_size = 500_000
    batch_size = 500

    execution_times_by_worker = {}
    
    for num_workers in num_workers_list:
        
        print(f"Running benchmark with {num_workers} workers (batch size: {batch_size})")
        execution_times = []
        
        for execution in range(num_executions):
            print(f"  Execution {execution+1}/{num_executions}")

            output_folder, log_folder = create_results_and_log_folders(output_folder, suffix=f"{num_workers}_{execution}")

            config_path = create_config(num_workers, batch_size)
            args = Namespace(
                shards_path=shards_path,
                ray_address=None,
                log_folder=str(log_folder),
                output_folder=output_folder,
                config_path=str(config_path)
            )
            
            took_sec = main(args)            
            execution_times.append(took_sec)
            print(f"Took {took_sec:.2f} seconds")
        
        # Calculate statistics
        mean_time = np.mean(execution_times)
        std_dev = np.std(execution_times)
        execution_times_by_worker[num_workers] = {
            "execution_times": execution_times,
            "mean_time": mean_time,
            "std_dev": std_dev
        }
        
        print(f"Average: {mean_time:.2f}s, Std Dev: {std_dev:.2f}s")
    


    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

