from pathlib import Path
import ray
import logging
import fasttext
from itertools import chain
from datetime import datetime

from made.config import Config
from made.paths import MADE_PATH
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import apply_filtering_step
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote#(num_gpus=1)
def ray_unimodal_text_filtering(tar_files: list[str | Path], log_folder: Path, config_path: Path):
    _ = Config(config_path)
    _ = MetricsStore()
    return unimodal_text_filtering(tar_files, log_folder)

def unimodal_text_filtering(tar_files: list[str | Path], log_folder: Path):
    logger = logging.getLogger("ray")

    # logger.info("Validating configuration")
    _validate_configuration()
    
    # TODO: cache model in a Ray Actor
    language_detection_model = fasttext.load_model(str(MADE_PATH / Config().unimodal.lang_detection_model_path)) 
    
    # logger.info("Decoding webdataset")
    dataset = decode_webdataset(
        tar_files,
        get_images=False,
        get_captions=True,
        batch_size=Config().unimodal.batch_size
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
        # TODO: implement this


        # ------------------------------------------- 
        # second step: filter by language
        ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filtering_step(
            filter_name=_get_filter_captions_by_language_mask,
            batch_id=batch_id,
            uids=batch[0],
            samples=batch[1],
            parameters = {
                "model": language_detection_model,
                "target_language": Config().unimodal.lang_detection_language,
                "threshold": Config().unimodal.lang_detection_score_threshold
            }
        )


        # ------------------------------------------- 
        # third step: pos tags filtering
        # TODO: implement this


        # ------------------------------------------- 
        # fourth step: text specificity filtering
        # TODO: implement this


        all_uids.append(ok_uids)

    # logger.info("Concatenating uids")
    all_uids = list(chain.from_iterable(all_uids))
    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if Config().infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)
    return all_uids

def _get_filter_captions_by_language_mask(
        captions: list[str],
        model: fasttext.FastText,
        target_language: str,
        threshold: float
    ) -> list[bool]:
    
    captions = [sentence.strip().replace('\n', ' ') for sentence in captions]

    predictions, scores = model.predict(captions)

    return [
        (pred == f"__label__{target_language}") and (score > threshold)
        for pred, score in zip(predictions, scores)
    ]
    

def _validate_configuration():
    config = Config()
    if config.unimodal.lang_detection_language not in ["en", "it", "es"]:
        raise ValueError("The language detection language must be either 'en' or 'it' or 'es'")
    if config.unimodal.lang_detection_score_threshold < 0.1 or config.unimodal.lang_detection_score_threshold > 1.0:
        raise ValueError("The language threshold must be between 0.1 and 1.0")
    if config.unimodal.batch_size <= 0:
        raise ValueError("The batch size must be greater than 0")
    if config.unimodal.lang_detection_model_path is None:
        raise ValueError("The fasttext model path must be provided")
