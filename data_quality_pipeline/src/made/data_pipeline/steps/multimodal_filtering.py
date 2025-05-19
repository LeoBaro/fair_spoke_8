from pathlib import Path
import logging
import os
import ray
import json
import torch
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from itertools import chain
from datetime import datetime
from collections import defaultdict

from transformers import CLIPProcessor, CLIPModel

from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import apply_filtering_step, FilteringBlock
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote(num_gpus=0.1)
class MultimodalFilter(FilteringBlock):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config = Config(config_path)
        _validate_configuration(self.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise ValueError("Multimodal filtering is not supported on CPU")
        os.environ["TOKENIZERS_PARALLELISM="] = "false"
        self.model = CLIPModel.from_pretrained(self.config.multimodal.clip_model).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.config.multimodal.clip_model) #  use_fast=True

    def execute(self, tar_files: list[str | Path], log_folder: Path, uids: list[str] = None):
        _ = MetricsStore()
        return multimodal_filtering(
            self.model,
            self.processor,
            tar_files, 
            log_folder, 
            self.config,
            uids
        )


def multimodal_filtering(
        clip_model,
        clip_processor,
        tar_files: list[str | Path],
        log_folder: Path, 
        config: Config,
        uids: list[str] = None
    ):
    logger = logging.getLogger("ray")
    
    # logger.info("Decoding webdataset")
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=config.multimodal.batch_size,
        valid_uids=uids
    )   
    
    # logger.info("Iterating over dataset")
    ok_uids = []
    filtered_uids_by_filter = defaultdict(list)

    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    while True:
        batch = get_next_batch(dataset_iter)
        if batch is None:
            break

        batch_id += 1
        sample_count += len(batch[0])
        # logger.info(f"Next batch {batch_id} / {sample_count}")

        # ------------------------------------------------------------------------ 
        # first step: filter by aspect ratio
        batch_ok_uids, batch_ok_samples, batch_uids_filtered, batch_samples_filtered = apply_filtering_step(
            filter_name=_get_clip_score_filter_mask,
            batch_id=batch_id,
            uids=batch[0],
            samples=batch,
            apply_filters=config.infrastructure.apply_filters,
            parameters = {
                "clip_model": clip_model,
                "clip_processor": clip_processor,
                "clip_score_threshold": config.multimodal.clip_score_threshold,
                "clip_caption_max_length": config.multimodal.clip_caption_max_length
            }
        )
        if config.infrastructure.save_filtered_uids:
            filtered_uids_by_filter["text_detection"].extend(batch_uids_filtered)

        # ------------------------------------------- 
        # third step: image specificity filtering
        # TODO: implement this


        ok_uids.append(batch_ok_uids)


    # logger.info("Concatenating uids")
    ok_uids = list(chain.from_iterable(ok_uids))

    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)

    if config.infrastructure.save_filtered_uids:
        filtered_uids_path = log_folder / "multimodal_filtering__filtered_uids_by_step.json"
        with open(filtered_uids_path, 'w', encoding="utf-8") as f:
            json.dump(filtered_uids_by_filter, f, indent=2)
        logger.info("Filtered UIDs saved to %s", filtered_uids_path)

    return ok_uids


def _get_clip_score_filter_mask(
        batch: list[Image.Image],
        clip_model,
        clip_processor,
        clip_score_threshold: float,
        clip_caption_max_length: int
    ) -> list[bool]:
    """
    Filter the images by aspect ratio.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    similarity_scores = []
    image_list = batch[1]
    text_list = batch[2]
    
    for img, txt in zip(image_list, text_list):
        inputs = clip_processor(
            text=[txt],
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=clip_caption_max_length
        ).to(device)
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.item()
        similarity_scores.append(score)

    return [
        (score > clip_score_threshold)
        for score in similarity_scores
    ]

def _validate_configuration(config: Config):
    if config.multimodal.clip_score_threshold < 0.0 or config.multimodal.clip_score_threshold > 1.0:
        raise ValueError("The aspect ratio threshold must be between 0.0 and 1.0")
