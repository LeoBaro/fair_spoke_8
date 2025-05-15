import logging
import os
import time
import numpy as np
import torch
import pprint
import ray
from pathlib import Path
from functools import partial
from transformers import CLIPModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from made.semdedup.compute_pretrained_embeddings import get_embeddings
from made.semdedup.dataloader import TarImageDataset, custom_collate_fn
from made.semdedup.clustering.clustering import compute_centroids
from made.semdedup.clustering.sort_clusters import assign_and_sort_clusters
from made.semdedup.semdedup_logic import process_shard
from made.semdedup.extract_dedup_data import extract_pruned_data
from made.data_pipeline.steps.base import apply_filtering_step, FilteringBlock
from made.paths import MADE_PATH
from made.config import Config


@ray.remote(num_gpus=0.1)
class SemDeDupFilter(FilteringBlock):
    def __init__(self, config_path: Path):
        self.config = Config(config_path)

    def execute(self, tar_files: list[str | Path], log_folder: Path):
        return run_complete_pipeline(tar_files, self.config)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Pipeline Function ---
def run_complete_pipeline(
        tar_files_directory: str,
        config: Config
):
    """
    Runs the entire SemDeDup pipeline sequentially within a single process.
    """
    start_time = time.time()
    logger.info("Starting the SemDeDup pipeline...")

    # # --- Load Configurations (Once) ---
    # NOTE: This is not needed because the config is passed as an argument
    # try:
    #     config = Config(MADE_PATH / "config.yaml")
    #     logger.info("Configurations loaded.")
    #     pp = pprint.PrettyPrinter(indent=4)
    #     logger.info("Main Config:")
    #     pp.pprint(config)

    # except FileNotFoundError as e:
    #     logger.error(f"Configuration file not found: {e}. Exiting.")
    #     return
    # except Exception as e:
    #     logger.error(f"Error loading configuration: {e}. Exiting.")
    #     return

    # --- Stage 1: Generate Embeddings ---
    try:
        logger.info("--- Stage 1: Generating Embeddings ---")
        stage_start_time = time.time()

        model_name = config.unimodal.semdedup.clip_model
        batch_size = config.unimodal.semdedup.batch_size
        paths_str_type = config.unimodal.semdedup.paths_str_type
        embed_float_type = config.unimodal.semdedup.embed_float_type
        emb_memory_loc = config.unimodal.semdedup.embs_memory_loc
        paths_memory_loc = config.unimodal.semdedup.path_memory_loc
        emb_size = config.unimodal.semdedup.emd_size

        os.makedirs(os.path.dirname(emb_memory_loc), exist_ok=True)
        os.makedirs(os.path.dirname(paths_memory_loc), exist_ok=True)

        logger.info("Loading model and image processor...")
        model = CLIPModel.from_pretrained(model_name)
        image_processor = CLIPImageProcessor.from_pretrained(model_name)

        logger.info("Setting up dataset and dataloader...")
        dataset = TarImageDataset(tar_dir=tar_files_directory, transform=None)
        my_collate_fn = partial(custom_collate_fn, image_processor=image_processor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=my_collate_fn,
            num_workers=config.unimodal.semdedup.num_workers
        )
        dataset_size = len(dataset)
        config.unimodal.semdedup.dataset_size = dataset_size

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

        logger.info("Computing embeddings...")
        
        get_embeddings(
            model, 
            dataloader, 
            emb_array, 
            path_array
        )

        # Flush forces any changes in the memory-mapped arrays to be written to disk
        # This ensures all embedding and path data is saved before closing the memmap files
        emb_array.flush()
        path_array.flush()
        del emb_array
        del path_array
        del model, image_processor, dataloader, dataset
        if torch.cuda.is_available():
             torch.cuda.empty_cache()

        logger.info(f"Stage 1 finished in {time.time() - stage_start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error in Stage 1 (Embeddings): {e}", exc_info=True)
        return

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
            data=emb_memory,
            ncentroids=config.unimodal.semdedup.clustering.num_clusters,
            niter=config.unimodal.semdedup.clustering.niter,   
            seed=config.unimodal.semdedup.clustering.seed,
            Kmeans_with_cos_dist=config.unimodal.semdedup.clustering.Kmeans_with_cos_dist,
            save_folder=config.unimodal.semdedup.clustering.save_folder,
            logger=logger,
            verbose=True
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

        # Reload memmaps for reading
        # emb_memory = np.memmap(
        #     emb_memory_loc,
        #     dtype=embed_float_type,
        #     mode='r',
        #     shape=(dataset_size, emb_size)
        # )

        paths_memory = np.memmap(
            paths_memory_loc,
            dtype=paths_str_type,
            mode='r',
            shape=(config['dataset_size'],)
        )

        assign_and_sort_clusters(
            data = emb_memory,
            paths_list = paths_memory,
            sim_metric = config.unimodal.semdedup.clustering.sim_metric,
            keep_hard = config.unimodal.semdedup.clustering.keep_hard,
            kmeans_with_cos_dist = config.unimodal.semdedup.clustering.Kmeans_with_cos_dist,
            save_folder = config.unimodal.semdedup.clustering.save_folder,
            sorted_clusters_file_loc = config.unimodal.semdedup.sorted_clusters_path,
            cluster_ids = range(0, config.unimodal.semdedup.clustering.num_clusters),
            logger=logger
        )
        del emb_memory
        del paths_memory

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

        extract_pruned_data(
            config.unimodal.semdedup.sorted_clusters_path,
            config.unimodal.semdedup.semdedup_pruning_tables_path,
            config.unimodal.semdedup.eps,
            config.unimodal.semdedup.clustering.num_clusters,
            config.unimodal.semdedup.output_txt_path,
            retreive_kept_samples=config.get('retreive_kept_samples', True)
        )

        logger.info(f"Stage 5 finished in {time.time() - stage_start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error in Stage 5 (Extract Data): {e}", exc_info=True)
        return

    # --- Pipeline Complete ---
    total_time = time.time() - start_time
    logger.info(f"--- Pipeline finished successfully in {total_time:.2f} seconds ({total_time/60:.2f} minutes) ---")


if __name__ == "__main__":
    run_complete_pipeline()