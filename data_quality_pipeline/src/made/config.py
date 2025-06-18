import os
import yaml
from pathlib import Path

from made.data_pipeline.common.singleton import Singleton
from made.data_pipeline.common.dict_wrapper import DictWrapper

class Config(metaclass=Singleton):

    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)
        for key, value in self._config.items():
            setattr(self, key, DictWrapper(value) if isinstance(value, dict) else value)

    def __str__(self):
        return yaml.dump(self._config, default_flow_style=False)
    
    @staticmethod
    def create_config(override_config: dict, output_path: Path):
        config_raw = f"""
infrastructure:
    enable_metrics: {override_config["enable_metrics"] if "enable_metrics" in override_config else True}
    save_bad_uids:  {override_config["save_bad_uids"] if "save_bad_uids" in override_config else True}
    logging_level:  {override_config["logging_level"] if "logging_level" in override_config else "INFO"}
    log_to_driver:  {override_config["log_to_driver"] if "log_to_driver" in override_config else True}
    num_workers:    {override_config["num_workers"] if "num_workers" in override_config else 1} 
    num_gpus_per_worker: {override_config["num_gpus_per_worker"] if "num_gpus_per_worker" in override_config else 0.1}
    batch_size:          {override_config["batch_size"] if "batch_size" in override_config else 50} 
    dump_tar_every_n_samples: {override_config["dump_tar_every_n_samples"] if "dump_tar_every_n_samples" in override_config else 10000}
    
unimodal_text:
    caption_min_words: {override_config["caption_min_words"] if "caption_min_words" in override_config else 2}
    caption_min_chars: {override_config["caption_min_chars"] if "caption_min_chars" in override_config else 5}

    lang_detection_model_path:      {override_config["lang_detection_model_path"] if "lang_detection_model_path" in override_config else "models/lid.176.bin"}
    lang_detection_score_threshold: {override_config["lang_detection_score_threshold"] if "lang_detection_score_threshold" in override_config else 0.7}
    lang_detection_language:        {override_config["lang_detection_language"] if "lang_detection_language" in override_config else "en"}

    tagging_model_name:                  {override_config["tagging_model_name"] if "tagging_model_name" in override_config else "en_core_web_trf"}
    good_captions_pos_distribution_path: {override_config["good_captions_pos_distribution_path"] if "good_captions_pos_distribution_path" in override_config else "models/common_pos_patterns.txt"}

unimodal_vision:
    image_min_aspect_ratio: {override_config["image_min_aspect_ratio"] if "image_min_aspect_ratio" in override_config else 0.8}
    image_max_aspect_ratio: {override_config["image_max_aspect_ratio"] if "image_max_aspect_ratio" in override_config else 3.0}
    image_min_dimension:    {override_config["image_min_dimension"] if "image_min_dimension" in override_config else 50}

    text_threshold:            {override_config["text_threshold"] if "text_threshold" in override_config else 0.6}
    text_detection_model_path: {override_config["text_detection_model_path"] if "text_detection_model_path" in override_config else "models"}
    text_detection_mag_ratio:  {override_config["text_detection_mag_ratio"] if "text_detection_mag_ratio" in override_config else 0.5}

multimodal:
    dfn_model:                         {override_config["dfn_model"] if "dfn_model" in override_config else "leobaro/DFN-public"}
    dfn_similarity_score_threshold:    {override_config["dfn_similarity_score_threshold"] if "dfn_similarity_score_threshold" in override_config else 4.8}
    clip_caption_max_length:           {override_config["clip_caption_max_length"] if "clip_caption_max_length" in override_config else 77}

specificity:
    model_path: {override_config["model_path"] if "model_path" in override_config else "models/ckpt.pt"}
    reference_path: {override_config["reference_path"] if "reference_path" in override_config else "models/reference.pt"}
    specificity_threshold: {override_config["specificity_threshold"] if "specificity_threshold" in override_config else 0.27}
    weight: {override_config["weight"] if "weight" in override_config else 0.5}
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(config_raw)
        return output_path