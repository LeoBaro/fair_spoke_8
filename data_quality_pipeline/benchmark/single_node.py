import os
import time
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import Namespace

os.environ["RAY_DEDUP_LOGS"] = "0" 


def create_config(num_workers, batch_size):
    config_raw = f"""
infrastructure:
    num_workers: {num_workers} 
    enable_metrics: false
    logging_level: WARNING
    save_npy: false

unimodal:
    batch_size: {batch_size} 

    caption_min_length: 10
    caption_max_length: 100

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 1.8
"""
    config_file_path = "/tmp/single_node_config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write(config_raw)
    return config_file_path

if __name__=='__main__':

    # batch size = 30000 is greater than the total number of samples split (25K)    

    from made.bin.made import main

    log_folder = Path(__file__).parent / "logs"
    log_folder.mkdir(exist_ok=True)
    output_folder = Path(__file__).parent / "out"
    output_folder.mkdir(exist_ok=True)
    results_file = Path(__file__).parent / "benchmark_results.json"

    num_executions = 3 

    results = {}

    num_workers_list = [2, 4, 8, 16, 32] 
    
    num_workers_baseline = num_workers_list[0] 
    batch_size = 500_000
    batch_size = 500

    execution_times_by_worker = {}
    
    for num_workers in num_workers_list:
        print(f"Running benchmark with {num_workers} workers (batch size: {batch_size})")
        execution_times = []
        
        for execution in range(num_executions):
            print(f"  Execution {execution+1}/{num_executions}")
            config_path = create_config(num_workers, batch_size)
            args = Namespace(
                shards_path="/davinci-1/work/lbaroncelli/datacomp_32_tar_files",
                ray_address=None,
                log_folder=str(log_folder),
                output_folder=output_folder,
                config_path=str(config_path)
            )
            
            took_sec = main(args)            
            execution_times.append(took_sec)
            print(f"    Took {took_sec:.2f} seconds")
        
        # Calculate statistics
        mean_time = np.mean(execution_times)
        std_dev = np.std(execution_times)
        execution_times_by_worker[num_workers] = {
            "execution_times": execution_times,
            "mean_time": mean_time,
            "std_dev": std_dev
        }
        
        print(f"  Average: {mean_time:.2f}s, Std Dev: {std_dev:.2f}s")
    
