import ray
import time
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from PIL import Image

from made.config import Config
from made.data_pipeline.common.metrics_store import MetricsStore
from made.data_pipeline.common.filtering_result import FilteringResult
from made.data_pipeline.base.filter_step import FilterStep
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

class BaseFilteringBlock(ABC):
    """Base class for all filtering blocks with common filtering logic"""

    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        self.logger = logging.getLogger("ray")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        if self.device == "cpu":
            raise ValueError("Filtering block is not supported on CPU")
        self.config = Config(config_path)
        self.log_folder = log_folder
        self.filtering_result = FilteringResult(output_folder, self.config.infrastructure.dump_tar_every_n_samples)
        self.metrics_store = MetricsStore(self.log_folder)
        self.worker_id = ray.get_runtime_context().get_worker_id() if ray.is_initialized() else None
        self.output_folder = Path(output_folder)

    @abstractmethod
    def get_filter_steps(self) -> List[FilterStep]:
        """Return list of filter steps to apply"""
        pass
    
    @abstractmethod
    def validate_configuration(self):
        """Validate configuration specific to this filter"""
        pass
    
    def apply_single_filter(self, 
        filter_step: FilterStep,
        uids: List[str], 
        captions: List[str], 
        images: List[Image.Image]) -> Tuple[List[str], List[str], List[Image.Image]]:
        
        """Apply a single filter step and return filtered data"""
        
        start_time = time.time()
        
        # Determine what inputs the filter function needs
        filter_func = filter_step.func
        params = filter_step.params.copy()
        
        # Call filter function with appropriate arguments
        if 'captions' in filter_func.__code__.co_varnames and 'images' in filter_func.__code__.co_varnames:
            mask = filter_func(captions, images, **params)
        elif 'captions' in filter_func.__code__.co_varnames:
            mask = filter_func(captions, **params)
        elif 'images' in filter_func.__code__.co_varnames:
            mask = filter_func(images, **params)
        else:
            raise ValueError(f"Filter function {filter_step.name} has unexpected signature")
        
        elapsed_time = time.time() - start_time
        
        # Record metrics
        if self.config.infrastructure.enable_metrics:
            self.metrics_store.add_filter_metric(
                filter_step.name,
                len(uids),
                sum(mask),
                elapsed_time,
                params,
                filter_step.param_keys_for_metrics
            )
        
        # Save bad UIDs if configured
        if self.config.infrastructure.save_bad_uids:
            bad_uids = [uid for uid, keep in zip(uids, mask) if not keep]
            self.metrics_store.dump_bad_uids(filter_step.name.replace('_get_', '').replace('_mask', ''), bad_uids)
        
        # Apply mask to filter data
        filtered_uids = [uid for uid, keep in zip(uids, mask) if keep]
        filtered_captions = [caption for caption, keep in zip(captions, mask) if keep]
        filtered_images = [image for image, keep in zip(images, mask) if keep]
        
        return filtered_uids, filtered_captions, filtered_images
    
    def execute(self, tar_files: List[str | Path]):
        """Common execution logic for all filtering blocks"""
        
        self.validate_configuration()
        
        dataset = decode_webdataset(
            tar_files,
            get_images=True,
            get_captions=True,
            batch_size=self.config.infrastructure.batch_size
        )
        
        sample_count = 0
        batch_id = 0
        dataset_iter = iter(dataset)
        filter_steps = self.get_filter_steps()
        
        self.logger.info("Starting %s filtering with %d steps", self.__class__.__name__, len(filter_steps))
        start_time = time.time()
        
        while True:
            batch_start_time = time.time()
            
            batch = get_next_batch(dataset_iter)
            if batch is None:
                break
            
            batch_id += 1
            sample_count += len(batch[0])
            
            # Start with all samples
            current_uids = batch[0]
            current_images = batch[1]
            current_captions = batch[2]
            
            # Apply each filter step sequentially
            for filter_step in filter_steps:
                if not current_uids:  # No samples left to filter
                    break
                    
                current_uids, current_captions, current_images = self.apply_single_filter(
                    filter_step, current_uids, current_captions, current_images
                )
            
            # Add remaining samples to results
            if current_uids:
                self.filtering_result.add_samples(current_uids, current_captions, current_images)
            
            self.filtering_result.dump_to_disk()
            
            batch_elapsed_time = time.time() - batch_start_time    
            self.logger.debug("Batch %s processed in %0.2f seconds", batch_id, batch_elapsed_time)
        
        self.filtering_result.dump_to_disk(force=True)
        
        elapsed_time = time.time() - start_time    
        self.logger.info("Total samples processed: %s in %0.2f seconds", sample_count, elapsed_time)
        
        if self.config.infrastructure.enable_metrics:
            self.metrics_store.save_to_file()
        
        return self.filtering_result.produced_tar_files, self.filtering_result.produced_uids_files