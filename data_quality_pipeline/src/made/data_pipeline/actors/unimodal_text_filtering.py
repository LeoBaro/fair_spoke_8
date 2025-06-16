from pathlib import Path
from typing import List

import ray
import spacy
import fasttext

from made.paths import MADE_PATH
from made.data_pipeline.base.base_filtering_block import BaseFilteringBlock, FilterStep
from made.data_pipeline.filtering_functions.text_filters import filter_captions_by_length, filter_captions_by_language, filter_captions_by_pos_tags

@ray.remote
class UnimodalTextFilter(BaseFilteringBlock):
    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        super().__init__(config_path, log_folder, output_folder)
        self.logger.info("Initializing UnimodalTextFilter on %s", self.device)
        
        # Load models
        self.language_detection_model = fasttext.load_model(
            str(MADE_PATH / self.config.unimodal_text.lang_detection_model_path)
        )
        spacy.require_gpu()
        self.tagging_model = spacy.load(self.config.unimodal_text.tagging_model_name)
        
        with open(str(MADE_PATH / self.config.unimodal_text.good_captions_pos_distribution_path), 'r') as file:
            self.common_pos_patterns = [line.strip() for line in file.readlines()]
    
    def get_filter_steps(self) -> List[FilterStep]:
        return [
            FilterStep(
                name="filter_captions_by_length",
                func=filter_captions_by_length,
                params={
                    "min_words": self.config.unimodal_text.caption_min_words,
                    "min_chars": self.config.unimodal_text.caption_min_chars,
                },
                param_keys_for_metrics=["min_words", "min_chars"]
            ),
            FilterStep(
                name="filter_captions_by_language",
                func=filter_captions_by_language,
                params={
                    "model": self.language_detection_model,
                    "target_language": self.config.unimodal_text.lang_detection_language,
                    "threshold": self.config.unimodal_text.lang_detection_score_threshold,
                },
                param_keys_for_metrics=["target_language", "threshold"]
            ),
            FilterStep(
                name="filter_captions_by_pos_tags",
                func=filter_captions_by_pos_tags,
                params={
                    "model": self.tagging_model,
                    "target_pos_tags": self.common_pos_patterns,
                },
                param_keys_for_metrics=[]
            ),
        ]
    
    def validate_configuration(self):
        if self.config.unimodal_text.lang_detection_language not in ["en", "it", "es"]:
            raise ValueError("The language detection language must be either 'en' or 'it' or 'es'")
        if not (0.1 <= self.config.unimodal_text.lang_detection_score_threshold <= 1.0):
            raise ValueError("The language threshold must be between 0.1 and 1.0")
        if self.config.unimodal_text.lang_detection_model_path is None:
            raise ValueError("The fasttext model path must be provided")

