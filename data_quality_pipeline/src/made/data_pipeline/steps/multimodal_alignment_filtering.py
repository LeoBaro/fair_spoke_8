from pathlib import Path
import logging
import os
import ray
import json
import torch
import numpy as np
import time
from PIL import Image
from itertools import chain, compress
from datetime import datetime
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel


from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import execute_filter, FilteringBlock, FilteringResult
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote
class MultimodalAlignmentFilter(FilteringBlock):
    def __init__(self, config_path: Path, log_folder: Path, webdataset_output_folder: Path):
        super().__init__(config_path, log_folder, webdataset_output_folder)
        self.logger.info("Initializing MultimodalFilter on %s", self.device)
        self.logger.info("Number of workers: %s", self.config.infrastructure.num_workers)
        self.logger.info("Batch size: %s", self.config.infrastructure.batch_size)
        self.logger.info("Log folder: %s", self.log_folder)
        self.logger.info("Output folder: %s", self.filtering_result.output_folder)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = CLIPModel.from_pretrained(self.config.multimodal.dfn_model).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.multimodal.dfn_model, use_fast=False) #  use_fast=True

    def execute(self, tar_files: list[str | Path]):
        return multimodal_alignment_filtering(
            tar_files, 
            self.model,
            self.processor,
            self.config,
            self.metrics_store,
            self.filtering_result
        )


def multimodal_alignment_filtering(
        tar_files: list[str | Path],
        dfn_model,
        clip_processor,
        config: Config,
        metrics_store: MetricsStore,
        filtering_result: FilteringResult
    ):
    logger = logging.getLogger("ray")
    

    _validate_configuration(config)

    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=config.infrastructure.batch_size
    )   
    
    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    logger.info("Starting multimodal filtering")
    start_time = time.time()

    while True:
        batch_start_time = time.time()
        
        batch = get_next_batch(dataset_iter)
        if batch is None:
            break

        batch_id += 1
        sample_count += len(batch[0])

        # ------------------------------------------------------------------------ 
        # first step: filter by dnf
        good_uids = batch[0]
        good_images = batch[1]
        good_captions = batch[2]

        filter_fn_parameters = {
            "dfn_model": dfn_model,
            "clip_processor": clip_processor,
            "dfn_percentile_to_drop": config.multimodal.dfn_percentile_to_drop,
            "clip_caption_max_length": config.multimodal.clip_caption_max_length
        }
        dfn_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_dfn_score_filter_mask,
            captions=good_captions,
            images=good_images,
            parameters = filter_fn_parameters
        )

        if config.infrastructure.enable_metrics:
            metrics_store.add_filter_metric(
                "_get_dfn_score_filter_mask",
                len(good_images),
                int(sum(dfn_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["dfn_percentile_to_drop", "clip_caption_max_length"]
            )
        if config.infrastructure.save_bad_uids:
            bad_uids = list(compress(good_uids, [not m for m in dfn_filter_mask]))
            metrics_store.dump_bad_uids("dfn", bad_uids)

        good_uids = list(compress(good_uids, [m for m in dfn_filter_mask]))
        good_images = list(compress(good_images, [m for m in dfn_filter_mask]))
        good_captions = list(compress(good_captions, [m for m in dfn_filter_mask]))

        filtering_result.add_samples(
            list(compress(good_uids, [m for m in dfn_filter_mask])), 
            list(compress(good_captions, [m for m in dfn_filter_mask])), 
            list(compress(good_images, [m for m in dfn_filter_mask]))
        )

        filtering_result.dump_to_disk()

        batch_elapsed_time = time.time() - batch_start_time    
        logger.info("Batch %s processed in %0.2f seconds", batch_id, batch_elapsed_time)

    filtering_result.dump_to_disk(force=True)

    elapsed_time = time.time() - start_time
    logger.info("Total samples processed: %s in %0.2f seconds", sample_count, elapsed_time)

    if config.infrastructure.enable_metrics:
        metrics_store.save_to_file()

    return filtering_result.produced_tar_files, filtering_result.produced_uids_files

def _get_dfn_score_filter_mask(
        captions: list[str],
        images: list[Image.Image],
        dfn_model,
        clip_processor,
        dfn_percentile_to_drop: int,
        clip_caption_max_length: int
    ) -> list[bool]:
    """
    Filter the images by aspect ratio.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    similarity_scores = []

    for img, txt in zip(images, captions):
        inputs = clip_processor(
            text=[txt],
            images=[img],
            return_tensors="pt",
            input_data_format="channels_last",
            padding=True,
            truncation=True,
            max_length=clip_caption_max_length
        ).to(device)
        outputs = dfn_model(**inputs)
        score = outputs.logits_per_image.item()
        similarity_scores.append(score)
        # except Exception as e:
        #     print(f"Error clip_processor: {e}")
        #     similarity_scores.append(0)
        #     continue
    
    return _filter_by_percentile(similarity_scores, dfn_percentile_to_drop)

def _filter_by_percentile(scores, percentile):
    """
    Returns a boolean mask indicating which samples to keep based on percentile filtering.

    Parameters:
    - scores (list of float): The list of scores (can be positive or negative).
    - percentile (int): Percentile value (0-100). Scores below this percentile will be discarded.

    Returns:
    - list of bool: Boolean mask, True if the sample should be kept, False otherwise.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    threshold = np.percentile(scores, percentile)
    mask = [score > threshold for score in scores]
    return mask

def _validate_configuration(config: Config):
    if config.multimodal.dfn_percentile_to_drop < 0 or config.multimodal.dfn_percentile_to_drop > 100:
        raise ValueError("The DFN percentile threshold must be between 0 and 100")
