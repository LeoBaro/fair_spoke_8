from pathlib import Path
import ray
import logging
import fasttext
import spacy
import json 
from itertools import chain, compress
from collections import defaultdict
from datetime import datetime
import torch
import numpy as np
from made.config import Config
from made.paths import MADE_PATH
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import execute_filter, FilteringBlock
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote(num_gpus=0.1)
class UnimodalTextFilter(FilteringBlock):

    def __init__(self, config_path: Path):
        super().__init__()
        self.config = Config(config_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise ValueError("UnimodalTextFilter filtering is not supported on CPU")
        self.language_detection_model = fasttext.load_model(
            str(MADE_PATH / self.config.unimodal_text.lang_detection_model_path)
            )
        self.tagging_model = spacy.load(self.config.unimodal_text.tagging_model_name)
        with open(
            str(MADE_PATH / self.config.unimodal_text.good_captions_pos_distribution_path),
            'r'
            ) as file:
            self.common_pos_patterns = [line.strip() for line in file.readlines()]
    
    def execute(self, tar_files: list[str | Path], log_folder: Path, uids: list[str] = None):
        _ = MetricsStore()
        return unimodal_text_filtering(
            self.language_detection_model,
            self.tagging_model,
            self.common_pos_patterns,
            tar_files, 
            log_folder, 
            self.config,
            uids
        )


def unimodal_text_filtering(
        language_detection_model,
        pos_tagging_model,
        pos_distribution,
        tar_files: list[str | Path], 
        log_folder: Path, 
        config: Config,
        uids: list[str] = None
    ):
    
    logger = logging.getLogger("ray")

    # logger.info("Validating configuration")
    _validate_configuration(config)
    
    # logger.info("Decoding webdataset")
    dataset = decode_webdataset(
        tar_files,
        get_images=False,
        get_captions=True,
        batch_size=config.unimodal_text.batch_size,
        valid_uids=uids
    )   

    # logger.info("Iterating over dataset")
    all_good_uids = []
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


        # ------------------------------------------- 
        # first step: filter by caption length
        good_uids = batch[0]
        good_captions = batch[1]

        filter_fn_parameters = {
            "min_words": config.unimodal_text.caption_min_words,
            "min_chars": config.unimodal_text.caption_min_chars
        }
        length_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_filter_captions_by_length_mask,
            captions=batch[1],
            images=None,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            MetricsStore().add_filter_metric(
                "_get_filter_captions_by_length_mask",
                batch_id,
                len(good_captions),
                int(sum(length_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["min_words", "min_chars"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in length_filter_mask]))
            filtered_uids_by_filter["length"].extend(bad_uids)

        good_uids = list(compress(good_uids, [m for m in length_filter_mask]))
        good_captions = list(compress(good_captions, [m for m in length_filter_mask]))

        # ------------------------------------------- 
        # second step: filter by language
        filter_fn_parameters = {
                "model": language_detection_model,
                "target_language": config.unimodal_text.lang_detection_language,
                "threshold": config.unimodal_text.lang_detection_score_threshold
        }
        lang_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_filter_captions_by_language_mask,
            captions=good_captions,
            images=None,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            MetricsStore().add_filter_metric(
                "_get_filter_captions_by_language_mask",
                batch_id,
                len(good_captions),
                int(sum(lang_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["model", "target_language", "threshold"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in lang_filter_mask]))
            filtered_uids_by_filter["language"].extend(bad_uids)

        good_uids = list(compress(good_uids, [m for m in lang_filter_mask]))
        good_captions = list(compress(good_captions, [m for m in lang_filter_mask]))

        # ------------------------------------------- 
        # third step: pos tags filtering
        filter_fn_parameters = {
            "model": pos_tagging_model,
            "target_pos_tags": pos_distribution
        }
        pos_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_filter_captions_by_pos_tags_mask,
            captions=good_captions,
            images=None,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            MetricsStore().add_filter_metric(
                "_get_filter_captions_by_pos_tags_mask",
                batch_id,
                len(good_captions),
                int(sum(pos_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["model"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in pos_filter_mask]))
            filtered_uids_by_filter["pos_tags"].extend(bad_uids)

        good_uids = list(compress(good_uids, [m for m in pos_filter_mask]))


        all_good_uids.append(good_uids)



    # logger.info("Concatenating uids")
    all_good_uids = list(chain.from_iterable(all_good_uids))

    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)

    if config.infrastructure.save_filtered_uids:
        worker_id = ray.get_runtime_context().get_worker_id() if ray.is_initialized() else "local"
        filtered_uids_path = log_folder / f"bad_uids_unimodal_text_filtering_{worker_id}.json"
        with open(filtered_uids_path, 'w', encoding="utf-8") as f:
            json.dump(filtered_uids_by_filter, f, indent=2)
        logger.info("Filtered UIDs saved to %s", filtered_uids_path)

    return all_good_uids

# TODO: implement a function than clean the captions and 
# remove extra whitespace and newlines

def _get_filter_captions_by_length_mask(
        captions: list[str],
        min_words: int,
        min_chars: int
    ) -> list[bool]:
    
    cleaned_captions = [
        caption.strip().replace('\n', ' ') for caption in captions
        ]
    
    tot_words = [len(caption.split()) for caption in cleaned_captions]
    tot_chars = [len(caption) for caption in cleaned_captions]

    # Create mask checking both conditions for each caption
    mask = [
        tot_words >= min_words and tot_chars >= min_chars
        for tot_words, tot_chars in zip(tot_words, tot_chars)
    ]

    return mask
        
def _get_filter_captions_by_language_mask(
        captions: list[str],
        model: fasttext.FastText,
        target_language: str,
        threshold: float
    ) -> list[bool]:
    
    captions = [sentence.strip().replace('\n', ' ') for sentence in captions]

    predictions, scores = model.predict(captions)

    return [
        (pred[0] == f"__label__{target_language}") and (score[0] > threshold)
        for pred, score in zip(predictions, scores)
    ]

def _get_filter_captions_by_pos_tags_mask(
        captions: list[str],
        model: spacy.Language,
        target_pos_tags: list[str]
    ) -> list[bool]:

    mask = []

    for caption in captions:
        doc = model(caption)
        pos_tags = [token.pos_ for token in doc]
        sorted_unique_tags = sorted(set(pos_tags))
        pos_string = "_".join(sorted_unique_tags)
        is_good_pos = pos_string in target_pos_tags
        mask.append(is_good_pos)
    
    return mask

def _validate_configuration(config: Config):
    if config.unimodal_text.lang_detection_language not in ["en", "it", "es"]:
        raise ValueError("The language detection language must be either 'en' or 'it' or 'es'")
    if config.unimodal_text.lang_detection_score_threshold < 0.1 or config.unimodal_text.lang_detection_score_threshold > 1.0:
        raise ValueError("The language threshold must be between 0.1 and 1.0")
    if config.unimodal_text.batch_size <= 0:
        raise ValueError("The batch size must be greater than 0")
    if config.unimodal_text.lang_detection_model_path is None:
        raise ValueError("The fasttext model path must be provided")
