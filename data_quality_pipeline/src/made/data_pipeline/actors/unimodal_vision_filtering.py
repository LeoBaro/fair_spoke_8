from pathlib import Path
from typing import List

import ray
import easyocr

from made.data_pipeline.base.base_filtering_block import BaseFilteringBlock, FilterStep
from made.data_pipeline.filtering_functions.vision_filters import filter_images_by_aspect_ratio, filter_images_by_text_detection

@ray.remote  
class UnimodalVisionFilter(BaseFilteringBlock):
    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        super().__init__(config_path, log_folder, output_folder)
        self.logger.info("Initializing UnimodalVisionFilter on %s", self.device)
        self.reader = easyocr.Reader(
            ['en'], 
            gpu=True, 
            user_network_directory=self.config.unimodal_vision.text_detection_model_path
        )
    
    def get_filter_steps(self) -> List[FilterStep]:
        return [
            FilterStep(
                name="filter_images_by_aspect_ratio",
                func=filter_images_by_aspect_ratio,
                params={
                    "image_min_aspect_ratio": self.config.unimodal_vision.image_min_aspect_ratio,
                    "image_max_aspect_ratio": self.config.unimodal_vision.image_max_aspect_ratio,
                    "image_min_dimension": self.config.unimodal_vision.image_min_dimension,
                },
                param_keys_for_metrics=["image_min_aspect_ratio", "image_max_aspect_ratio", "image_min_dimension"]
            ),
            FilterStep(
                name="filter_images_by_text_detection", 
                func=filter_images_by_text_detection,
                params={
                    "model": self.reader,
                    "text_thresh": self.config.unimodal_vision.text_threshold,
                    "mag_ratio": self.config.unimodal_vision.text_detection_mag_ratio,
                },
                param_keys_for_metrics=["text_thresh", "mag_ratio"]
            ),
        ]
    
    def validate_configuration(self):
        if not (0.0 <= self.config.unimodal_vision.image_min_aspect_ratio <= 1.0):
            raise ValueError("The aspect ratio threshold must be between 0.0 and 1.0")