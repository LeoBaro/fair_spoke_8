import os
from pathlib import Path
from typing import List

import ray
import torch
from transformers import CLIPTokenizer

from made.paths import MADE_PATH
from made.data_pipeline.base.base_filtering_block import BaseFilteringBlock, FilterStep
from made.data_pipeline.filtering_functions.multimodal_filters import filter_by_specificity
from made.models.meru.nn import model_init


@ray.remote
class MultimodalSpecificityFilter(BaseFilteringBlock):
    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        super().__init__(config_path, log_folder, output_folder)
        self.logger.info("Initializing MultimodalSpecificityFilter on %s", self.device)
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.meru_model, self.trs = model_init(pretrained=str(MADE_PATH / self.config.specificity.model_path))
        self.meru_model = self.meru_model.to(self.device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        references = torch.load(str(MADE_PATH / self.config.specificity.reference_path))
        self.img_ref = references["img"].to(self.device)
        self.txt_ref = references["txt"].to(self.device)
    
    def get_filter_steps(self) -> List[FilterStep]:
        return [
            FilterStep(
                name="filter_by_specificity",
                func=filter_by_specificity,
                params={
                    "meru_model": self.meru_model,
                    "trs": self.trs,
                    "tokenizer": self.tokenizer,
                    "img_ref": self.img_ref,
                    "txt_ref": self.txt_ref,
                    "curvature": self.meru_model.curvature.exp(),
                    "specificity_threshold": self.config.specificity.specificity_threshold,
                    "weight": self.config.specificity.weight,
                },
                param_keys_for_metrics=["specificity_threshold", "weight"]
            ),
        ]
    
    def validate_configuration(self):
        if not (self.config.specificity.specificity_threshold > 0):
            raise ValueError("The specificity threshold must be greater than 0")
        if not (self.config.specificity.weight >= 0 and self.config.specificity.weight <= 1):
            raise ValueError("The weight must be between 0 and 1")