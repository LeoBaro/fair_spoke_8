from pathlib import Path
from typing import List

import ray
from transformers import CLIPModel, CLIPProcessor

from made.data_pipeline.base.base_filtering_block import BaseFilteringBlock, FilterStep
from made.data_pipeline.filtering_functions.multimodal_filters import filter_by_clip_similarity


@ray.remote
class MultimodalAlignmentFilter(BaseFilteringBlock):
    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        super().__init__(config_path, log_folder, output_folder)
        self.logger.info("Initializing MultimodalAlignmentFilter on %s", self.device)
        
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = CLIPModel.from_pretrained(self.config.multimodal.dfn_model).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.multimodal.dfn_model, use_fast=False)
    
    def get_filter_steps(self) -> List[FilterStep]:
        return [
            FilterStep(
                name="filter_by_clip_similarity",
                func=filter_by_clip_similarity,
                params={
                    "dfn_model": self.model,
                    "clip_processor": self.processor,
                    "dfn_similarity_score_threshold": self.config.multimodal.dfn_similarity_score_threshold,
                    "clip_caption_max_length": self.config.multimodal.clip_caption_max_length,
                },
                param_keys_for_metrics=["dfn_similarity_score_threshold", "clip_caption_max_length"]
            ),
        ]
    
    def validate_configuration(self):
        if not (self.config.multimodal.dfn_similarity_score_threshold > 0):
            raise ValueError("The DFN similarity score threshold must be greater than 0")