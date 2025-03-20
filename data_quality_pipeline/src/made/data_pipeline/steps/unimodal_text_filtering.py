from pathlib import Path
import ray
import logging
import fasttext
import spacy
from itertools import chain
from datetime import datetime

from made.config import Config
from made.paths import MADE_PATH
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import apply_filtering_step
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote
class UnimodalTextFilter:

    def __init__(self, config_path: Path):
        self.config = Config(config_path)
        self.language_detection_model = fasttext.load_model(
            str(MADE_PATH / self.config.unimodal.lang_detection_model_path)
            )
        self.tagging_model = spacy.load(
            str(MADE_PATH / self.config.unimodal.tagging_model_path)
            )
        with open(
            str(MADE_PATH / self.config.unimodal.good_captions_pos_distribution_path),
            'r'
            ) as file:
            self.common_pos_patterns = [line.strip() for line in file.readlines()]
    
    def ray_unimodal_text_filtering(self, tar_files: list[str | Path], log_folder: Path):
        _ = MetricsStore()
        return unimodal_text_filtering(
            self.language_detection_model,
            self.tagging_model,
            self.common_pos_patterns,
            tar_files, 
            log_folder, 
            self.config
        )


def unimodal_text_filtering(
        language_detection_model,
        pos_tagging_model,
        pos_distribution,
        tar_files: list[str | Path], 
        log_folder: Path, 
        config: Config
    ):
    
    logger = logging.getLogger("ray")

    # logger.info("Validating configuration")
    _validate_configuration(config)
    
    # logger.info("Decoding webdataset")
    dataset = decode_webdataset(
        tar_files,
        get_images=False,
        get_captions=True,
        batch_size=config.unimodal.batch_size
    )   

    # logger.info("Iterating over dataset")
    all_uids = []
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
        ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filtering_step(
            filter_name=_get_filter_captions_by_length_mask,
            batch_id=batch_id,
            uids=batch[0],
            samples=batch[1],
            pipeline_type= config.unimodal.pipeline_type,
            parameters = {
                "min_words": config.unimodal.caption_min_words,
                "min_chars": config.unimodal.caption_max_chars
            }
        )

        # ------------------------------------------- 
        # second step: filter by language
        ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filtering_step(
            filter_name=_get_filter_captions_by_language_mask,
            batch_id=batch_id,
            uids=ok_uids,
            samples=ok_samples,
            pipeline_type= config.unimodal.pipeline_type,
            parameters = {
                "model": language_detection_model,
                "target_language": config.unimodal.lang_detection_language,
                "threshold": config.unimodal.lang_detection_score_threshold
            }
        )


        # ------------------------------------------- 
        # third step: pos tags filtering
        ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filtering_step(
            filter_name = _get_filter_captions_by_pos_tags_mask,
            batch_id=batch_id,
            uids=ok_uids,
            samples=ok_samples,
            pipeline_type= config.unimodal.pipeline_type,
            parameters = {
                "model": pos_tagging_model,
                "target_pos_tags": pos_distribution
            }
        )

        # ------------------------------------------- 
        # fourth step: text specificity filtering
        # TODO: implement this


        all_uids.append(ok_uids)

    # logger.info("Concatenating uids")
    all_uids = list(chain.from_iterable(all_uids))
    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)
    return all_uids

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
    if config.unimodal.lang_detection_language not in ["en", "it", "es"]:
        raise ValueError("The language detection language must be either 'en' or 'it' or 'es'")
    if config.unimodal.lang_detection_score_threshold < 0.1 or config.unimodal.lang_detection_score_threshold > 1.0:
        raise ValueError("The language threshold must be between 0.1 and 1.0")
    if config.unimodal.batch_size <= 0:
        raise ValueError("The batch size must be greater than 0")
    if config.unimodal.lang_detection_model_path is None:
        raise ValueError("The fasttext model path must be provided")
