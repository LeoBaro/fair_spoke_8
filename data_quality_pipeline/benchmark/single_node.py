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

    num_workers_list = [1, 2, 4, 8, 16, 32] 
    num_workers_baseline = num_workers_list[0] 
    batch_size = 500_000

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
    


    # Save results
    with open(results_file, "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {}
        for w, metrics in execution_times_by_worker.items():
            serializable_results[str(w)] = {
                "execution_times": [float(t) for t in metrics["execution_times"]],
                "mean_time": float(metrics["mean_time"]),
                "std_dev": float(metrics["std_dev"])
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Prepare data for visualization
    plot_data = []
    for workers, metrics in execution_times_by_worker.items():
        for i, time_value in enumerate(metrics["execution_times"]):
            plot_data.append({
                "Workers": workers,
                "Execution": i+1,
                "Execution Time (s)": time_value
            })
    
    summary_data = []
    for workers, metrics in execution_times_by_worker.items():
        summary_data.append({
            "Workers": workers,
            "Mean Time (s)": metrics["mean_time"],
            "Std Dev": metrics["std_dev"]
        })
    
    df = pd.DataFrame(plot_data)
    summary_df = pd.DataFrame(summary_data)
    
    # Create visualizations
    plt.figure(figsize=(12, 15))
    
    # 1. Line plot: Workers vs Mean Execution Time
    plt.subplot(3, 1, 1)
    sns.lineplot(
        data=summary_df, 
        x="Workers", 
        y="Mean Time (s)", 
        marker='o'
    )
    plt.title("Effect of Worker Count on Execution Time")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylabel("Execution Time (s)")
    
    # # 2. Bar plot with individual runs
    # plt.subplot(3, 1, 2)
    # sns.barplot(
    #     data=df,
    #     x="Workers",
    #     y="Execution Time (s)",
    #     hue="Execution",
    #     errorbar=None
    # )
    # plt.title("Individual Run Times by Worker Count")
    # plt.legend(title="Execution #")
    
    # 3. Speedup analysis
    plt.subplot(3, 1, 3)
    
    # Calculate speedup relative to single worker
    baseline_time = execution_times_by_worker[num_workers_baseline]["mean_time"]
    speedup_data = []
    
    for workers in num_workers_list:
        current_time = execution_times_by_worker[workers]["mean_time"]
        speedup = baseline_time / current_time
        efficiency = speedup / workers
        speedup_data.append({
            "Workers": workers,
            "Speedup": speedup,
            "Efficiency": efficiency * 100  # As percentage
        })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Twin axes for speedup and efficiency
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot speedup
    sns.lineplot(
        data=speedup_df,
        x="Workers",
        y="Speedup",
        marker='o',
        color='blue',
        ax=ax1
    )
    
    # Plot ideal speedup
    x = np.array(num_workers_list)
    ax1.plot(x, x, 'k--', label="Ideal Speedup")
    
    # # Plot efficiency on secondary axis
    # sns.lineplot(
    #     data=speedup_df,
    #     x="Workers",
    #     y="Efficiency",
    #     marker='s',
    #     color='red',
    #     ax=ax2
    # )
    
    ax1.set_title("Speedup vs Number of Workers")
    ax1.set_xlabel("Number of Workers")
    ax1.set_ylabel("Speedup (relative to 1 worker)")
    
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "worker_scaling_results.png")
    
    print("Visualizations saved to worker_scaling_results.png")
