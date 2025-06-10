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
from made.data_pipeline.steps.base import execute_filter, FilteringBlock
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote(num_gpus=0.1)
class MultimodalFilter(FilteringBlock):
    def __init__(self, config_path: Path):
        super().__init__()
        self.logger.info("Initializing MultimodalFilter on %s", self.device)
        self.config = Config(config_path)
        _validate_configuration(self.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise ValueError("Multimodal filtering is not supported on CPU")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model = CLIPModel.from_pretrained(self.config.multimodal.dfn_model).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.config.multimodal.dfn_model, use_fast=False) #  use_fast=True

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
        dfn_model,
        clip_processor,
        tar_files: list[str | Path],
        log_folder: Path, 
        config: Config,
        uids: list[str] = None
    ):
    logger = logging.getLogger("ray")
    

    _validate_configuration(config)

    # logger.info("Decoding webdataset")
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=config.multimodal.batch_size,
        valid_uids=uids
    )   
    
    # logger.info("Iterating over dataset")
    all_good_uids = []
    filtered_uids_by_filter = defaultdict(list)

    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    logger.info("Starting multimodal filtering")
    start_time = time.time()
    while True:
        batch = get_next_batch(dataset_iter)
        if batch is None:
            break

        batch_id += 1
        sample_count += len(batch[0])
        # logger.info(f"Next batch {batch_id} / {sample_count}")

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
            MetricsStore().add_filter_metric(
                "_get_dfn_score_filter_mask",
                batch_id,
                len(batch[0]),
                int(sum(dfn_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["dfn_percentile_to_drop", "clip_caption_max_length"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in dfn_filter_mask]))
            filtered_uids_by_filter["dfn"].extend(bad_uids)

        good_uids = list(compress(good_uids, [m for m in dfn_filter_mask]))
        good_images = list(compress(good_images, [m for m in dfn_filter_mask]))
        good_captions = list(compress(good_captions, [m for m in dfn_filter_mask]))


        all_good_uids.append(good_uids)


    # logger.info("Concatenating uids")
    all_good_uids = list(chain.from_iterable(all_good_uids))

    elapsed_time = time.time() - start_time
    logger.info("Total samples processed: %s in %0.2f seconds", sample_count, elapsed_time)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)

    if config.infrastructure.save_filtered_uids:
        worker_id = ray.get_runtime_context().get_worker_id() if ray.is_initialized() else "local"
        filtered_uids_path = log_folder / f"bad_uids_multimodal_filtering_{worker_id}.json"
        with open(filtered_uids_path, 'w', encoding="utf-8") as f:
            json.dump(filtered_uids_by_filter, f, indent=2)
        logger.info("Filtered UIDs saved to %s", filtered_uids_path)

    return all_good_uids

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
