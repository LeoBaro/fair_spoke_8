import gc
import json
import os
import ray
import time
import logging
import torch
import numpy as np

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from itertools import compress
from transformers import CLIPModel, CLIPImageProcessor

from made.paths import MADE_PATH
from made.config import Config
from made.semdedup.compute_pretrained_embeddings import get_embeddings
from made.semdedup.clustering.clustering import compute_centroids
from made.semdedup.clustering.sort_clusters import assign_and_sort_clusters
from made.semdedup.semdedup_logic import process_shard
from made.semdedup.extract_dedup_data import extract_pruned_data
from made.data_pipeline.steps.base import execute_filter, FilteringBlock
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.data.datacomp_handler import (
    decode_webdataset, 
    get_next_batch,
    get_dataset_size
)

@ray.remote(num_gpus=0.1)
class SemDeDupFilter(FilteringBlock):
    def __init__(self, config_path: Path):
        self.config = Config(config_path)
        self.model = CLIPModel.from_pretrained(
            self.config.unimodal.semdedup.clip_model
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.config.unimodal.semdedup.clip_model
        )

    def execute(
            self, 
            tar_files: list[str | Path], 
            log_folder: Path, 
            uids: list[str] = None
        ):
        _ = MetricsStore()
        return semdedup_filtering(
            self.model,
            self.image_processor,
            tar_files, 
            log_folder,
            self.config,
            uids
            )

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def semdedup_filtering(
        model,
        image_processor,
        tar_files: list[str | Path],
        log_folder: Path,
        config: Config,
        uids: list[str] = None
):
    logger = logging.getLogger("ray")
    # start_time = time.time()
    logger.info("Starting the SemDeDup pipeline...")

    _validate_configuration(config)

    # -- Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # -- model 
    model = model.to(device)
    model = model.eval()
    
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=False,
        batch_size=config.unimodal.semdedup.batch_size,
        valid_uids=uids
    )

    all_good_uids = []
    filtered_uids_by_filter = defaultdict(list)

    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    try:
        dataset_size = get_dataset_size(dataset)
        config.unimodal.semdedup.dataset_size = dataset_size

        paths_str_type = config.unimodal.semdedup.paths_str_type
        embed_float_type = config.unimodal.semdedup.embed_float_type
        emb_memory_loc = config.unimodal.semdedup.embs_memory_loc
        paths_memory_loc = config.unimodal.semdedup.path_memory_loc
        emb_size = config.unimodal.semdedup.emd_size

        os.makedirs(os.path.dirname(emb_memory_loc), exist_ok=True)
        os.makedirs(os.path.dirname(paths_memory_loc), exist_ok=True)
        
        logger.info(f"Dataset size: {dataset_size}")
        logger.info("Initializing memmap arrays...")
        emb_array = np.memmap(
            emb_memory_loc, 
            dtype=embed_float_type,
            mode='w+',
            shape=(dataset_size, emb_size)
        )
        path_array = np.memmap(
            paths_memory_loc,
            dtype=paths_str_type,
            mode='w+', 
            shape=(dataset_size,)
            )
    except Exception as e:
        logger.error(f"Error in initialization and model loading: {e}", exc_info=True)
        return
    logger.info("--- Stage 1: Computing Embeddings ---")
    batch_size = config.unimodal.semdedup.batch_size
    dummy_filter_mask = [1] * batch_size

    while True:
        stage_start_time = time.time()
        batch = get_next_batch(dataset_iter)
        if batch is None:
            break

        batch_id += 1
        good_uids = batch[0]
        good_images = batch[1]
        sample_count += len(good_uids)

    
        batch_indices = np.arange(sample_count - len(good_uids), sample_count)

        filter_fn_parameters = {
            "model": model,
            "image_processor": image_processor,
            "batch_size": config.unimodal.semdedup.batch_size,
            "valid_uids": uids
        }

        inputs = image_processor(images = good_images, return_tensors = "pt")
        data_batch = inputs["pixel_values"].to('cuda')

        get_embeddings(
            model,
            data_batch,
            batch_indices,
            emb_array,
            path_array,
            good_uids
        )
        elapsed_time = time.time() - stage_start_time

        if config.infrastructure.enable_metrics:
            MetricsStore().add_filter_metric(
                "get_embeddings",
                batch_id,
                len(batch[0]),
                int(len(batch[0])),
                elapsed_time,
                filter_fn_parameters,
                ["batch_size", "valid_uids"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in dummy_filter_mask]))
            filtered_uids_by_filter["semdedup"].extend(bad_uids)


        good_uids = list(compress(good_uids, [m for m in dummy_filter_mask]))
        good_images = list(compress(good_images, [m for m in dummy_filter_mask]))

        all_good_uids.append(good_uids)

    logger.info(f"Stage 1 finished in {time.time() - stage_start_time:.2f} seconds.")
    # Flush forces any changes in the memory-mapped arrays to be written to disk
    # This ensures all embedding and path data is saved before closing the memmap files
    emb_array.flush()
    path_array.flush()
    del emb_array
    del path_array
    del model, image_processor, dataset
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    filter_fn_parameters = {
        "config": config,
        "dataset_size": dataset_size,
        "emb_size": emb_size,
        "log_folder": log_folder
    }

    semdedup_filter_mask, elapsed_time = execute_filter(
        filter_name=_get_semdedup_filter_mask,
        captions=None,
        images=None,
        parameters=filter_fn_parameters
    )
 
    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)

    if config.infrastructure.save_filtered_uids:
        filtered_uids_path = log_folder / "bad_uids_multimodal_filtering.json"
        with open(filtered_uids_path, 'w', encoding="utf-8") as f:
            json.dump(filtered_uids_by_filter, f, indent=2)
        logger.info("Filtered UIDs saved to %s", filtered_uids_path)

    return semdedup_filter_mask


def _get_semdedup_filter_mask(
        config: Config,
        dataset_size: int,
        emb_size: int,
        log_folder: Path,
):
    
    logger = logging.getLogger("ray")
    """
    Runs the entire SemDeDup pipeline sequentially within a single process.
    """
    start_time = time.time()

    # --- Stage 2: Clustering ---
    try:
        logger.info("--- Stage 2: Computing Centroids ---")
        stage_start_time = time.time()

        emb_memory = np.memmap(
            config.unimodal.semdedup.embs_memory_loc,
            dtype=config.unimodal.semdedup.embed_float_type,
            mode='r',
            shape=(dataset_size, emb_size)
        )

        compute_centroids(
            data = emb_memory,
            ncentroids = config.unimodal.semdedup.clustering.num_clusters,
            niter = config.unimodal.semdedup.clustering.niter,   
            seed = config.unimodal.semdedup.seed,
            Kmeans_with_cos_dist = config.unimodal.semdedup.clustering.Kmeans_with_cos_dist,
            save_folder = config.unimodal.semdedup.clustering.save_folder,
            logger = logger,
            verbose = True
        )
        # del emb_memory # Close memmap

        logger.info(f"Stage 2 finished in {time.time() - stage_start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error in Stage 2 (Clustering): {e}", exc_info=True)
        return

    # --- Stage 3: Sort Clusters ---
    try:
        logger.info("--- Stage 3: Assigning and Sorting Clusters ---")
        stage_start_time = time.time()

        paths_memory = np.memmap(
            config.unimodal.semdedup.path_memory_loc,
            dtype=config.unimodal.semdedup.paths_str_type,
            mode='r',
            shape=(dataset_size,)
        )

        assign_and_sort_clusters(
            data = emb_memory,
            uids_list = paths_memory,
            sim_metric = config.unimodal.semdedup.clustering.sim_metric,
            keep_hard = config.unimodal.semdedup.clustering.keep_hard,
            kmeans_with_cos_dist = config.unimodal.semdedup.clustering.Kmeans_with_cos_dist,
            save_folder = config.unimodal.semdedup.clustering.save_folder,
            sorted_clusters_file_loc = config.unimodal.semdedup.sorted_clusters_path,
            cluster_ids = range(0, config.unimodal.semdedup.clustering.num_clusters),
            logger=logger
        )
        del emb_memory, paths_memory

        logger.info(f"Stage 3 finished in {time.time() - stage_start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error in Stage 3 (Sort Clusters): {e}", exc_info=True)
        return

    # --- Stage 4: SemDeDup ---
    try:
        logger.info("--- Stage 4: Performing Semantic Deduplication ---")
        stage_start_time = time.time()
          
        process_shard(shard=0, config=config)

        logger.info(f"Stage 4 finished in {time.time() - stage_start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error in Stage 4 (SemDeDup): {e}", exc_info=True)
        return

    # --- Stage 5: Extract Duplicated Data ---
    try:
        logger.info("--- Stage 5: Extracting Pruned Data List ---")
        stage_start_time = time.time()

        all_good_uids = extract_pruned_data(
            config.unimodal.semdedup.sorted_clusters_path,
            config.unimodal.semdedup.semdedup_pruning_tables_path,
            config.unimodal.semdedup.eps,
            config.unimodal.semdedup.clustering.num_clusters,
            config.unimodal.semdedup.output_txt_path,
            retreive_kept_samples = getattr(config, 'retreive_kept_samples', True)
        )

        logger.info(f"Stage 5 finished in {time.time() - stage_start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error in Stage 5 (Extract Data): {e}", exc_info=True)
        return

    # --- Pipeline Complete ---
    total_time = time.time() - start_time
    logger.info(f"--- Pipeline finished successfully in {total_time:.2f} seconds ({total_time/60:.2f} minutes) ---")

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)
    
    return all_good_uids

def _validate_configuration(config: Config):
    if config.unimodal.semdedup.batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    if config.unimodal.semdedup.clustering.num_clusters < 1:
        raise ValueError("Number of clusters must be at least 1")
    if config.unimodal.semdedup.clustering.niter < 1:
        raise ValueError("Number of iterations must be at least 1")