import ray
import os
import gc
import time
import numpy as np
from pathlib import Path
from typing import List, Set
import torch
from transformers import CLIPModel, CLIPImageProcessor

from made.paths import MADE_PATH
from made.data_pipeline.base.base_filtering_block import BaseFilteringBlock
from made.semdedup.compute_pretrained_embeddings import get_embeddings
from made.semdedup.clustering.clustering import compute_centroids
from made.semdedup.clustering.sort_clusters import assign_and_sort_clusters
from made.semdedup.semdedup_logic import process_shard
from made.semdedup.extract_dedup_data import extract_pruned_data
from made.data_pipeline.data.datacomp_handler import (
    decode_webdataset, 
    get_next_batch,
    get_dataset_size
)


@ray.remote
class SemanticDedupFilter(BaseFilteringBlock):
    """
    Semantic deduplication filter using CLIP embeddings and clustering.
    
    This filter implements a 5-stage pipeline:
    1. Compute embeddings for all samples
    2. Perform clustering to find centroids
    3. Assign samples to clusters and sort
    4. Apply semantic deduplication logic
    5. Extract final list of samples to keep
    """
    
    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        super().__init__(config_path, log_folder, output_folder)
        self.logger.info("Initializing SemanticDedupFilter on %s", self.device)
        
        # Initialize CLIP model
        self.model = CLIPModel.from_pretrained(self.config.semdedup.clip_model)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config.semdedup.clip_model)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        
        # Store the final set of UIDs to keep
        self.kept_uids: Set[str] = set()
    
    def get_filter_steps(self) -> List:
        """Override to return empty list since we handle filtering differently"""
        return []
    
    def execute(self, tar_files: List[str | Path]):
        """Execute the 5-stage semantic deduplication pipeline"""
        
        self.validate_configuration()
        
        start_time = time.time()
        
        try:
            # First, get all UIDs from the dataset for validation
            initial_dataset = decode_webdataset(
                tar_files,
                get_images=False,
                get_captions=False,
                batch_size=self.config.infrastructure.batch_size
            )
            
            all_uids = []
            dataset_iter = iter(initial_dataset)
            while True:
                batch = get_next_batch(dataset_iter)
                if batch is None:
                    break
                all_uids.extend(batch[0])  # UIDs are first element
            
            # Run the 5-stage pipeline
            self._run_semdedup_pipeline(tar_files, all_uids)
            
            # Filter and save the final results
            self._filter_and_save_results(tar_files)
            
        except Exception as e:
            self.logger.error(f"Error in semantic deduplication pipeline: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_resources()
        
        total_time = time.time() - start_time
        self.logger.info(
            f"SemanticDedupFilter completed in {total_time:.2f} seconds "
            f"({total_time/60:.2f} minutes)"
        )
        
        if self.config.infrastructure.enable_metrics:
            self.metrics_store.save_to_file()
        
        return self.filtering_result.produced_tar_files, self.filtering_result.produced_uids_files
    
    def _run_semdedup_pipeline(self, tar_files: List[str | Path], all_uids: List[str]):
        """Execute the 5-stage semantic deduplication pipeline"""
        
        # Setup dataset with images for embedding computation
        dataset = decode_webdataset(
            tar_files,
            get_images=True,
            get_captions=False,
            batch_size=self.config.semdedup.batch_size,
            valid_uids=all_uids
        )
        
        dataset_size = len(all_uids)
        self.config.semdedup.dataset_size = dataset_size
        
        # Setup memory-mapped arrays
        self._setup_memmap_arrays(dataset_size)
        
        # Stage 1: Compute Embeddings
        self._stage1_compute_embeddings(dataset)
        
        # Stage 2: Compute Centroids
        self._stage2_compute_centroids(dataset_size)
        
        # Stage 3: Assign and Sort Clusters
        self._stage3_sort_clusters(dataset_size)
        
        # Stage 4: Semantic Deduplication
        self._stage4_semdedup()
        
        # Stage 5: Extract Final Results
        self._stage5_extract_results()
    
    def _setup_memmap_arrays(self, dataset_size: int):
        """Setup memory-mapped arrays for embeddings and paths"""
        
        paths_str_type = self.config.semdedup.paths_str_type
        embed_float_type = self.config.semdedup.embed_float_type
        emb_memory_loc = self.config.semdedup.embs_memory_loc
        paths_memory_loc = self.config.semdedup.path_memory_loc
        emb_size = self.config.semdedup.emd_size
        
        os.makedirs(os.path.dirname(emb_memory_loc), exist_ok=True)
        os.makedirs(os.path.dirname(paths_memory_loc), exist_ok=True)
        
        self.logger.info(f"Dataset size: {dataset_size}")
        self.logger.info("Initializing memmap arrays...")
        
        self.emb_array = np.memmap(
            emb_memory_loc, 
            dtype=embed_float_type,
            mode='w+',
            shape=(dataset_size, emb_size)
        )
        self.path_array = np.memmap(
            paths_memory_loc,
            dtype=paths_str_type,
            mode='w+', 
            shape=(dataset_size,)
        )
    
    def _stage1_compute_embeddings(self, dataset):
        """Stage 1: Compute CLIP embeddings for all samples"""
        
        self.logger.info("--- Stage 1: Computing Embeddings ---")
        stage_start_time = time.time()
        
        dataset_iter = iter(dataset)
        batch_size = self.config.semdedup.batch_size
        batch_indices = 0
        
        while True:
            batch = get_next_batch(dataset_iter)
            if batch is None:
                break
            
            uids, images, _ = batch
            good_images = [img for img in images if img is not None]
            good_uids = [uid for uid, img in zip(uids, images) if img is not None]
            
            if not good_images:
                continue
            
            # Process images through CLIP
            inputs = self.image_processor(images=good_images, return_tensors="pt")
            data_batch = inputs["pixel_values"].to(self.device)
            
            # Compute embeddings using semdedup function
            get_embeddings(
                self.model,
                data_batch,
                list(range(batch_indices, batch_indices + len(good_images))),
                self.emb_array,
                self.path_array,
                good_uids
            )
            
            batch_indices += len(good_images)
        
        # Flush and cleanup
        self.emb_array.flush()
        self.path_array.flush()
        del self.emb_array
        del self.path_array
        
        self.logger.info(f"Stage 1 finished in {time.time() - stage_start_time:.2f} seconds.")
    
    def _stage2_compute_centroids(self, dataset_size: int):
        """Stage 2: Compute clustering centroids"""
        
        self.logger.info("--- Stage 2: Computing Centroids ---")
        stage_start_time = time.time()
        
        try:
            emb_memory = np.memmap(
                self.config.semdedup.embs_memory_loc,
                dtype=self.config.semdedup.embed_float_type,
                mode='r',
                shape=(dataset_size, self.config.semdedup.emd_size)
            )
            
            compute_centroids(
                data=emb_memory,
                ncentroids=self.config.semdedup.clustering.num_clusters,
                niter=self.config.semdedup.clustering.niter,   
                seed=self.config.semdedup.seed,
                Kmeans_with_cos_dist=self.config.semdedup.clustering.Kmeans_with_cos_dist,
                save_folder=self.config.semdedup.clustering.save_folder,
                logger=self.logger,
                verbose=True
            )
            
            self.logger.info(f"Stage 2 finished in {time.time() - stage_start_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in Stage 2 (Clustering): {e}", exc_info=True)
            raise
    
    def _stage3_sort_clusters(self, dataset_size: int):
        """Stage 3: Assign samples to clusters and sort"""
        
        self.logger.info("--- Stage 3: Assigning and Sorting Clusters ---")
        stage_start_time = time.time()
        
        try:
            emb_memory = np.memmap(
                self.config.semdedup.embs_memory_loc,
                dtype=self.config.semdedup.embed_float_type,
                mode='r',
                shape=(dataset_size, self.config.semdedup.emd_size)
            )
            
            paths_memory = np.memmap(
                self.config.semdedup.path_memory_loc,
                dtype=self.config.semdedup.paths_str_type,
                mode='r',
                shape=(dataset_size,)
            )
            
            assign_and_sort_clusters(
                data=emb_memory,
                uids_list=paths_memory,
                sim_metric=self.config.semdedup.clustering.sim_metric,
                keep_hard=self.config.semdedup.clustering.keep_hard,
                kmeans_with_cos_dist=self.config.semdedup.clustering.Kmeans_with_cos_dist,
                save_folder=self.config.semdedup.clustering.save_folder,
                sorted_clusters_file_loc=self.config.semdedup.sorted_clusters_path,
                cluster_ids=range(0, self.config.semdedup.clustering.num_clusters),
                logger=self.logger
            )
            
            del emb_memory, paths_memory
            
            self.logger.info(f"Stage 3 finished in {time.time() - stage_start_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in Stage 3 (Sort Clusters): {e}", exc_info=True)
            raise
    
    def _stage4_semdedup(self):
        """Stage 4: Perform semantic deduplication"""
        
        self.logger.info("--- Stage 4: Performing Semantic Deduplication ---")
        stage_start_time = time.time()
        
        try:
            process_shard(shard=0, config=self.config)
            
            self.logger.info(f"Stage 4 finished in {time.time() - stage_start_time:.2f} seconds.")
            
        except Exception as e:
            self.logger.error(f"Error in Stage 4 (SemDeDup): {e}", exc_info=True)
            raise
    
    def _stage5_extract_results(self):
        """Stage 5: Extract final list of samples to keep"""
        
        self.logger.info("--- Stage 5: Extracting Pruned Data List ---")
        stage_start_time = time.time()
        
        try:
            all_good_uids = extract_pruned_data(
                self.config.semdedup.sorted_clusters_path,
                self.config.semdedup.semdedup_pruning_tables_path,
                self.config.semdedup.eps,
                self.config.semdedup.clustering.num_clusters,
                self.config.semdedup.output_txt_path,
                retreive_kept_samples=getattr(self.config, 'retreive_kept_samples', True)
            )
            
            # Store the UIDs to keep
            self.kept_uids = set(all_good_uids)
            
            self.logger.info(f"Stage 5 finished in {time.time() - stage_start_time:.2f} seconds.")
            self.logger.info(f"Semantic deduplication kept {len(self.kept_uids)} samples")
            
        except Exception as e:
            self.logger.error(f"Error in Stage 5 (Extract Data): {e}", exc_info=True)
            raise
    
    def _filter_and_save_results(self, tar_files: List[str | Path]):
        """Filter the original dataset and save only the kept samples"""
        
        self.logger.info("Filtering and saving final results...")
        
        # Create new dataset iterator for filtering
        dataset = decode_webdataset(
            tar_files,
            get_images=True,
            get_captions=True,
            batch_size=self.config.infrastructure.batch_size
        )
        
        dataset_iter = iter(dataset)
        total_processed = 0
        total_kept = 0
        
        while True:
            batch = get_next_batch(dataset_iter)
            if batch is None:
                break
            
            uids, images, captions = batch
            total_processed += len(uids)
            
            # Filter batch based on kept UIDs
            kept_indices = [i for i, uid in enumerate(uids) if uid in self.kept_uids]
            
            if kept_indices:
                kept_uids = [uids[i] for i in kept_indices]
                kept_images = [images[i] for i in kept_indices]
                kept_captions = [captions[i] for i in kept_indices]
                
                self.filtering_result.add_samples(kept_uids, kept_captions, kept_images)
                total_kept += len(kept_indices)
            
            # Periodically save to disk
            self.filtering_result.dump_to_disk()
        
        # Final save
        self.filtering_result.dump_to_disk(force=True)
        
        # Record metrics
        if self.config.infrastructure.enable_metrics:
            self.metrics_store.add_filter_metric(
                "semantic_deduplication",
                total_processed,
                total_kept,
                0,  # Time measured separately for each stage
                {
                    "num_clusters": self.config.semdedup.clustering.num_clusters,
                    "eps": self.config.semdedup.eps,
                    "dedup_ratio": total_kept / total_processed if total_processed > 0 else 0
                },
                ["num_clusters", "eps"]
            )
        
        self.logger.info(f"Final filtering: kept {total_kept}/{total_processed} samples "
                        f"({total_kept/total_processed*100:.1f}%)")
    
    def _cleanup_resources(self):
        """Clean up GPU memory and model resources"""
        
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'image_processor'):
            del self.image_processor
        
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def validate_configuration(self):
        """Validate semantic deduplication configuration"""
        
        if not hasattr(self.config, 'semdedup'):
            raise ValueError("Missing 'semdedup' configuration section")
        
        required_fields = [
            'clip_model', 'batch_size', 'embs_memory_loc', 'path_memory_loc',
            'emd_size', 'paths_str_type', 'embed_float_type', 'seed', 'eps'
        ]
        
        for field in required_fields:
            if not hasattr(self.config.semdedup, field):
                raise ValueError(f"Missing semdedup configuration field: {field}")
        
        if not hasattr(self.config.semdedup, 'clustering'):
            raise ValueError("Missing 'clustering' configuration section under semdedup")
        
        clustering_fields = [
            'num_clusters', 'niter', 'Kmeans_with_cos_dist', 'save_folder',
            'sim_metric', 'keep_hard'
        ]
        
        for field in clustering_fields:
            if not hasattr(self.config.semdedup.clustering, field):
                raise ValueError(f"Missing clustering configuration field: {field}")
        
        # Validate paths exist
        required_paths = [
            'sorted_clusters_path', 'semdedup_pruning_tables_path', 'output_txt_path'
        ]
        
        for path_field in required_paths:
            if not hasattr(self.config.semdedup, path_field):
                raise ValueError(f"Missing path configuration: {path_field}")