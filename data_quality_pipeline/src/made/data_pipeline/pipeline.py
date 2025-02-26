from pathlib import Path

import ray

from made.data_pipeline.steps.unimodal_text_filtering import ray_unimodal_text_filtering
from made.data_pipeline.steps.unimodal_vision_filtering import ray_unimodal_vision_filtering
#from made.data_pipeline.steps.multimodal_filtering import ray_multimodal_filtering
#from made.data_pipeline.steps.vision_deduplication import ray_vision_deduplication
from made.config import Config

ray.init()

def split_and_distribute_dataset(tar_files: list[str | Path], num_workers: int):
    """Distributes dataset shards across workers."""

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

    """
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
    """

    return None

if __name__ == "__main__":

    num_workers = Config().infrastructure.num_workers
    shards_path = "/home/leobaro/Downloads/datasets/web/datacomp/_shards_1"

    tar_files = sorted([str(shards_path/s) for s in Path(shards_path).glob("*.tar")])

    results = split_and_distribute_dataset(tar_files, num_workers)
    
    for res in results:
        print(res)  # Processed results from workers

    ray.shutdown()