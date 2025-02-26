import atexit
import argparse
from pathlib import Path

import ray
import numpy as np

from made.config import Config
from made.data_pipeline.steps.unimodal_text_filtering import ray_unimodal_text_filtering
from made.data_pipeline.steps.unimodal_vision_filtering import ray_unimodal_vision_filtering
#from made.data_pipeline.steps.multimodal_filtering import ray_multimodal_filtering
#from made.data_pipeline.steps.vision_deduplication import ray_vision_deduplication
from made.data_pipeline.utils import print_execution_stats
from made.data_pipeline.metrics.metrics_collector import MetricsCollector

ray.init()

def save_uids(uids: list[str], out_filename: str):
    """
    The format describing the subset of samples should be a numpy array of dtype 
    numpy.dtype("u8,u8") (i.e. a structured array of pairs of unsigned 64-bit integers), 
    with shape (subset_size,), containing a list of uids (128-bit hashes from the parquet 
    files) in lexicographic sorted order, saved to disk in either npy format or 
    memory-mapped format.    
    """
    processed_uids = np.array([(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids], np.dtype("u8,u8"))
    processed_uids.sort()
    np.save(out_filename, processed_uids)

def run_pipeline(
        tar_files: list[str | Path], 
        num_workers: int,
    ) -> list[str]:
    """
    This pipeline will select the subset of samples that will be used for CLIP training.
    Each sample in the dataset has a unique identifier, which is present in the metadata 
    parquets, and in the json files inside the .tar shards.
    """

    # Ensure each worker gets a subset of tar files
    shard_splits = [tar_files[i::num_workers] for i in range(num_workers)]
    

    # Launch Ray tasks
    text_filtering_futures = [
        ray_unimodal_text_filtering.remote(shards) for shards in shard_splits if shards
    ]
    vision_filtering_futures = [
        ray_unimodal_vision_filtering.remote(shards) for shards in shard_splits if shards
    ]

    unimodal_text_filtering_results = ray.get(text_filtering_futures)
    unimodal_vision_filtering_results = ray.get(vision_filtering_futures)

    # Concatenate results
    concatenated_text_results = sum(unimodal_text_filtering_results, [])
    concatenated_vision_results = sum(unimodal_vision_filtering_results, [])

    
    # Pass results to multimodal filtering along with the original input
    multimodal_filtering_futures = [
        ray_multimodal_filtering.remote(shards, concatenated_text_results, concatenated_vision_results) 
        for shards in shard_splits if shards
    ]

    multimodal_filtering_results = ray.get(multimodal_filtering_futures)

    vision_deduplication_futures = [
        ray_vision_deduplication.remote(shards, multimodal_filtering_results)
        for shards in shard_splits if shards
    ]

    vision_deduplication_results = ray.get(vision_deduplication_futures)
    

    return []


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_path", required=False, nargs="+",
                        default=["/home/leobaro/Downloads/datasets/web/datacomp/_shards_1"])
    return parser.parse_args()

def main(args):

    tar_files = sorted([str(s) for s in Path(args.shards_path).glob("*.tar")])

    if ray.is_initialized():
        metrics_collector = MetricsCollector.options(name="metrics_collector").remote()
    
    ok_uids = run_pipeline(tar_files, Config().infrastructure.num_workers)

    print(f"Got {len(ok_uids)} uids")
    save_uids(ok_uids, "ok_uids.npy")

    # Collect and save aggregated metrics
    if ray.is_initialized():
        metrics_collector = ray.get_actor("metrics_collector")
        metrics_path = ray.get(metrics_collector.save_aggregated_metrics.remote("logs/metrics"))
        print(f"Aggregated metrics saved to: {metrics_path}")
    
    print("Done")


    # TODO: After creating a subset, you may invoke the resharder to build the subset shards 
    # From: https://github.com/mlfoundations/datacomp
    # in $output_dir like so: 
    # python resharder.py -i $download_dir -o $output_dir -s $subset_file


    ray.shutdown()
    
    atexit.register(print_execution_stats)


if __name__ == "__main__":
    args = cli()
    main(args)