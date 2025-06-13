import time
import logging
from pathlib import Path
from itertools import compress
from collections import defaultdict

import ray
import spacy
import fasttext

from made.config import Config
from made.paths import MADE_PATH
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import execute_filter, FilteringBlock, FilteringResult
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote(num_gpus=0.1, max_concurrency=1)
class UnimodalTextFilter(FilteringBlock):

    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        super().__init__(config_path, log_folder, output_folder)
        self.logger.info("Initializing UnimodalTextFilter on %s", self.device)
        self.logger.info("Number of workers: %s", self.config.infrastructure.num_workers)
        self.logger.info("Batch size: %s", self.config.infrastructure.batch_size)
        self.logger.info("Log folder: %s", self.log_folder)
        self.logger.info("Output folder: %s", self.filtering_result.output_folder)

        self.language_detection_model = fasttext.load_model(
            str(MADE_PATH / self.config.unimodal_text.lang_detection_model_path)
            )
        spacy.require_gpu()        
        self.tagging_model = spacy.load(self.config.unimodal_text.tagging_model_name)
        with open(
            str(MADE_PATH / self.config.unimodal_text.good_captions_pos_distribution_path),
            'r'
            ) as file:
            self.common_pos_patterns = [line.strip() for line in file.readlines()]

    def execute(self, tar_files: list[str | Path]):
        return unimodal_text_filtering(
            tar_files, 
            self.language_detection_model,
            self.tagging_model,
            self.common_pos_patterns,
            self.config,
            self.metrics_store,
            self.filtering_result
        )


def unimodal_text_filtering(
        tar_files: list[str | Path],
        language_detection_model,
        pos_tagging_model,
        pos_distribution,
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
        batch_size=config.unimodal_text.batch_size
    )   

    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    logger.info("Starting unimodal text filtering")
    start_time = time.time()

    while True:
        batch_start_time = time.time()
        
        batch = get_next_batch(dataset_iter)
        if batch is None:
            break
        
        batch_id += 1
        sample_count += len(batch[0])
        # logger.info(f"Next batch {batch_id} / {sample_count}")

        # ------------------------------------------- 
        # first step: filter by caption length
        good_uids = batch[0]
        good_images = batch[1]
        good_captions = batch[2]

        filter_fn_parameters = {
            "min_words": config.unimodal_text.caption_min_words,
            "min_chars": config.unimodal_text.caption_min_chars
        }
        length_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_filter_captions_by_length_mask,
            captions=good_captions,
            images=None,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            metrics_store.add_filter_metric(
                "_get_filter_captions_by_length_mask",
                len(good_captions),
                int(sum(length_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["min_words", "min_chars"]
            )
        if config.infrastructure.save_bad_uids:
            metrics_store.dump_bad_uids("length", list(compress(good_uids, [not m for m in length_filter_mask])))

        good_uids = list(compress(good_uids, [m for m in length_filter_mask]))
        good_captions = list(compress(good_captions, [m for m in length_filter_mask]))
        good_images = list(compress(good_images, [m for m in length_filter_mask]))

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
            metrics_store.add_filter_metric(
                "_get_filter_captions_by_language_mask",
                len(good_captions),
                int(sum(lang_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["model", "target_language", "threshold"]
            )
        if config.infrastructure.save_bad_uids:
            metrics_store.dump_bad_uids("language", list(compress(good_uids, [not m for m in lang_filter_mask])))


        good_uids = list(compress(good_uids, [m for m in lang_filter_mask]))
        good_captions = list(compress(good_captions, [m for m in lang_filter_mask]))
        good_images = list(compress(good_images, [m for m in lang_filter_mask]))

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
            metrics_store.add_filter_metric(
                "_get_filter_captions_by_pos_tags_mask",
                len(good_captions),
                int(sum(pos_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["model"]
            )
        if config.infrastructure.save_bad_uids:
            metrics_store.dump_bad_uids("pos_tags", list(compress(good_uids, [not m for m in pos_filter_mask])))

        filtering_result.add_samples(
            list(compress(good_uids, [m for m in pos_filter_mask])), 
            list(compress(good_captions, [m for m in pos_filter_mask])), 
            list(compress(good_images, [m for m in pos_filter_mask]))
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
    if config.unimodal_text.lang_detection_model_path is None:
        raise ValueError("The fasttext model path must be provided")
