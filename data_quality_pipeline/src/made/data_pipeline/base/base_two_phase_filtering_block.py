from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np
from pathlib import Path

from made.data_pipeline.base.base_filtering_block import BaseFilteringBlock
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch


class BaseTwoPhaseFilteringBlock(BaseFilteringBlock):
    """Base class for filtering blocks that require two phases: collection and filtering"""
    
    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        super().__init__(config_path, log_folder, output_folder)
        self.collected_data = []  # Store all samples during collection phase
        self.global_state = {}    # Store computed global information (e.g., centroids)
    
    @abstractmethod
    def process_sample_for_collection(self, uid: str, caption: str, image) -> Any:
        """Process a single sample during collection phase. Return data to store."""
        pass
    
    @abstractmethod
    def compute_global_state(self) -> None:
        """Compute global state (e.g., centroids) from collected data"""
        pass
    
    @abstractmethod
    def should_keep_sample(self, sample_data: Any) -> bool:
        """Decide whether to keep a sample based on global state"""
        pass
    
    def get_filter_steps(self) -> List:
        """Override to return empty list since we don't use the standard filtering pipeline"""
        return []
    
    def execute(self, tar_files: List[str | Path]):
        """Two-phase execution: collect, then filter"""
        
        self.validate_configuration()
        
        # Phase 1: Collection
        self.logger.info("Phase 1: Collecting data and computing embeddings...")
        self._collection_phase(tar_files)
        
        # Phase 2: Compute global state
        self.logger.info("Phase 2: Computing global state (clustering)...")
        self.compute_global_state()
        
        # Phase 3: Filtering
        self.logger.info("Phase 3: Filtering samples based on global state...")
        self._filtering_phase()
        
        self.filtering_result.dump_to_disk(force=True)
        
        if self.config.infrastructure.enable_metrics:
            self.metrics_store.save_to_file()
        
        return self.filtering_result.produced_tar_files, self.filtering_result.produced_uids_files
    
    def _collection_phase(self, tar_files: List[str | Path]):
        """Collect all samples and process them for global analysis"""
        dataset = decode_webdataset(
            tar_files,
            get_images=True,
            get_captions=True,
            batch_size=self.config.infrastructure.batch_size
        )
        
        dataset_iter = iter(dataset)
        sample_count = 0
        
        while True:
            batch = get_next_batch(dataset_iter)
            if batch is None:
                break
            
            uids, images, captions = batch
            sample_count += len(uids)
            
            # Process each sample in the batch
            for uid, caption, image in zip(uids, captions, images):
                sample_data = self.process_sample_for_collection(uid, caption, image)
                self.collected_data.append({
                    'uid': uid,
                    'caption': caption,
                    'image': image,
                    'processed_data': sample_data
                })
            
            self.logger.debug("Collected %d samples so far", len(self.collected_data))
        
        self.logger.info("Collection phase completed. Total samples: %d", sample_count)
    
    def _filtering_phase(self):
        """Filter samples based on computed global state"""
        kept_count = 0
        
        for sample in self.collected_data:
            if self.should_keep_sample(sample['processed_data']):
                self.filtering_result.add_samples(
                    [sample['uid']], 
                    [sample['caption']], 
                    [sample['image']]
                )
                kept_count += 1
            
            # Periodically dump to disk to manage memory
            if kept_count % self.config.infrastructure.dump_tar_every_n_samples == 0:
                self.filtering_result.dump_to_disk()
        
        self.logger.info("Filtering phase completed. Kept %d out of %d samples", 
                        kept_count, len(self.collected_data))