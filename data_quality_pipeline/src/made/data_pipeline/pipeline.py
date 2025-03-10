from pathlib import Path

import ray
import logging

from made.data_pipeline.steps.unimodal_text_filtering import UnimodalTextFilter
from made.data_pipeline.steps.unimodal_vision_filtering import UnimodalVisionFilter
#from made.data_pipeline.steps.multimodal_filtering import ray_multimodal_filtering
#from made.data_pipeline.steps.vision_deduplication import ray_vision_deduplication


def run_pipeline(
        tar_files: list[str | Path], 
        num_workers: int,
        log_folder: str | Path,
        config_path: str | Path
    ) -> list[str]:
    """
    This pipeline will select the subset of samples that will be used for CLIP training.
    Each sample in the dataset has a unique identifier, which is present in the metadata 
    parquets, and in the json files inside the .tar tar.
    """

    logger = logging.getLogger("ray")
    logger.info("Running pipeline")

    # Creating Ray actors (an actor is essentially a stateful worker)
    unimodal_text_filtering_actors = [
        UnimodalTextFilter.remote(config_path) for _ in range(num_workers)
    ]    

    unimodal_vision_filtering_actors = [
        UnimodalVisionFilter.remote(config_path) for _ in range(num_workers)
    ]    


    # Ensure each worker gets a subset of tar files
    tar_splits = [tar_files[i::num_workers] for i in range(num_workers)]
    logger.info(f"Num. workers: {num_workers}. Shard split: {[len(s) for s in tar_splits]}")


    # Launch Ray tasks
    text_filtering_futures = [
        actor.ray_unimodal_text_filtering.remote(tar_split, log_folder) for actor, tar_split in zip(unimodal_text_filtering_actors, tar_splits) if tar_split
    ]

    vision_filtering_futures = [
        actor.ray_unimodal_vision_filtering.remote(tar_split, log_folder) for actor, tar_split in zip(unimodal_vision_filtering_actors, tar_splits) if tar_split
    ]

    unimodal_text_filtering_results = ray.get(text_filtering_futures)
    unimodal_vision_filtering_results = ray.get(vision_filtering_futures)

    # Concatenate results
    # concatenated_text_results = sum(unimodal_text_filtering_results, [])
    # concatenated_vision_results = sum(unimodal_vision_filtering_results, [])

    
    # Pass results to multimodal filtering along with the original input
    # multimodal_filtering_futures = [
    #     ray_multimodal_filtering.remote(tar, concatenated_text_results, concatenated_vision_results) 
    #     for tar in tar_splits if tar
    # ]

    # multimodal_filtering_results = ray.get(multimodal_filtering_futures)

    # vision_deduplication_futures = [
    #     ray_vision_deduplication.remote(tar, multimodal_filtering_results)
    #     for tar in tar_splits if tar
    # ]

    # vision_deduplication_results = ray.get(vision_deduplication_futures)
    

    return []