from pathlib import Path

import ray
import fasttext
import numpy as np
from numpy.typing import NDArray

from made.config import Config
from made.data_pipeline.metrics.metrics_decorators import get_time
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.metrics.utils import send_metrics_central_collector
from made.data_pipeline.utils import apply_filter_mask
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from made.paths import MADE_PATH

@ray.remote
def ray_unimodal_text_filtering(tar_files: list[str | Path]):
    results = unimodal_text_filtering(tar_files)
    send_metrics_central_collector()
    return results

@get_time
def unimodal_text_filtering(tar_files: list[str | Path]):
    """
    This is the first step of the data quality pipeline, hence it gets the data directly from the tar files.
    """
    _validate_configuration()

    language_detection_model = fasttext.load_model(str(MADE_PATH / Config().unimodal.lang_detection_model_path)) # TODO: cache model in a Ray Actor
    
    
    dataset = decode_webdataset(
        tar_files,
        get_images=False,
        get_captions=True,
        batch_size=Config().unimodal.batch_size
    )   
    
    all_uids = []
    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    while True:
        batch = get_next_batch(dataset_iter)
        if batch is None:
            break
        
        uids = batch[0]
        captions = [sentence.strip().replace('\n', ' ') for sentence in batch[1]]
        batch_id += 1
        sample_count += len(uids)


        # ------------------------------------------- 
        # first step: filter by caption length
        # TODO: implement this


        # ------------------------------------------- 
        # second step: filter by language
        filter_mask: list[bool] 
        filter_mask = _get_filter_captions_by_language_mask(
            language_detection_model,
            captions,
            Config().unimodal.lang_detection_language,
            Config().unimodal.lang_detection_score_threshold
        )
        ok_uids, ok_captions, uids_filtered, captions_filtered = apply_filter_mask(
            uids, captions, filter_mask,
            filter_name="_get_filter_captions_by_language_mask",
            batch_id=batch_id
        )


        # ------------------------------------------- 
        # third step: pos tags filtering
        # TODO: implement this


        # ------------------------------------------- 
        # fourth step: text specificity filtering
        # TODO: implement this


        all_uids.append(ok_uids)


    all_uids = np.concatenate(all_uids)

    return all_uids

@get_time
def _get_filter_captions_by_language_mask(
        model: fasttext.FastText,
        captions: list[str],
        target_language: str,
        threshold: float
    ) -> NDArray[np.bool_]:
    
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
