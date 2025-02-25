from pathlib import Path

from made.data_pipeline.data.datacomp_handler import decode_webdataset
from made.data_pipeline.config import Config
from made.paths import MADE_PATH
from made.data_pipeline.utils import get_time

import ray
import fasttext
import numpy as np
from numpy.typing import NDArray

@ray.remote
def ray_unimodal_text_filtering(tar_files: list[str | Path]):
    return unimodal_text_filtering(tar_files)

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

    # The tar 0000000 of the first shard contains 10033 elements and the batch size is 12000, hence only one iteration is performed
    for batch in dataset:

        uids = batch[0]
        captions = [sentence.strip().replace('\n', ' ') for sentence in batch[1]]

        # ------------------------------------------- 
        # first step: filter by caption length
        # TODO: implement this


        # ------------------------------------------- 
        # second step: filter by language
        filter_mask: NDArray[np.bool_]
        filter_mask = _get_filter_captions_by_language_mask(
            language_detection_model,
            captions,
            Config().unimodal.lang_detection_language,
            Config().unimodal.lang_detection_score_threshold
        )

        ok_uids: NDArray[np.str_]
        ok_captions: NDArray[np.str_]
        uids_filtered: NDArray[np.str_]
        captions_filtered: NDArray[np.str_]
        ok_uids, ok_captions, uids_filtered, captions_filtered = _apply_filter_mask(uids, captions, filter_mask)
        # _log_results() # TODO: implement this



        # ------------------------------------------- 
        # third step: pos tags filtering
        # TODO: implement this
        breakpoint()

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

    return np.array([
        (pred == f"__label__{target_language}") and (score > threshold)
        for pred, score in zip(predictions, scores)
    ], dtype=np.bool_)

@get_time
def _apply_filter_mask(
        uids: list[str],
        captions: list[str],
        filter_mask: NDArray[np.bool_]
    ) -> tuple[NDArray[np.str_], NDArray[np.str_]]:
    """
    Apply the filter mask to the uids and captions.
    Return 4 arrays:    
    - uids: the uids that passed the filter
    - captions: the captions that passed the filter
    - uids_filtered: the uids that did not pass the filter
    - captions_filtered: the captions that did not pass the filter
    """
    return np.array(uids)[filter_mask], np.array(captions)[filter_mask], np.array(uids)[~filter_mask], np.array(captions)[~filter_mask]

@get_time
def _save_results(results: list[tuple[str, str]]):
    with open(Config().unimodal.output_path, "w") as f:
        for uid, caption in results:
            f.write(f"{uid}\t{caption}\n")

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
