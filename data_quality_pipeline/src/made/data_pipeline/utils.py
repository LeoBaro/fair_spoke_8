from typing import Any
from pathlib import Path

import ray
import numpy as np
from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

def collect_tar_files(shards_path: list[str | Path]):
    return sorted([str(s) for s in Path(shards_path).glob("*.tar")])

def get_worker_id():
    return ray.get_runtime_context().get_worker_id() if ray.is_initialized() else "local"

def shutdown_ray():
    if ray.is_initialized():
        ray.shutdown()  

def print_execution_stats():
    """Print execution statistics and filter metrics summaries"""
    summary = MetricsStore().get_summary()
    
    print("\n===== Execution Time Statistics =====")
    for func_name, stats in summary["execution_times"].items():
        print(f"{func_name}:")
        print(f"  Mean = {stats['mean']:.4f}s, StdDev = {stats['stddev']:.4f}s")
        print(f"  Min = {stats['min']:.4f}s, Max = {stats['max']:.4f}s, Calls = {stats['calls']}")
    
    print("\n===== Filter Metrics Summary =====")
    for func_name, stats in summary["filter_metrics"].items():
        print(f"{func_name}:")
        print(f"  Total input: {stats['total_input']}, Total output: {stats['total_output']}")
        print(f"  Total filtered: {stats['total_filtered']} ({stats['avg_filter_rate']*100:.2f}%)")
        print(f"  Batches processed: {stats['batches_processed']}")
    
    # Save metrics to file
    summary_path, details_path = MetricsStore().save_to_file(Config().infrastructure.log_folder)
    print(f"\nMetrics saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Details: {details_path}")

def save_uids(uids: list[str], output_folder: str | Path):
    """
    The format describing the subset of samples should be a numpy array of dtype 
    numpy.dtype("u8,u8") (i.e. a structured array of pairs of unsigned 64-bit integers), 
    with shape (subset_size,), containing a list of uids (128-bit hashes from the parquet 
    files) in lexicographic sorted order, saved to disk in either npy format or 
    memory-mapped format.    
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_filename = output_folder / "ok_uids.npy"
    processed_uids = np.array([(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids], np.dtype("u8,u8"))
    processed_uids.sort()
    np.save(out_filename, processed_uids)    