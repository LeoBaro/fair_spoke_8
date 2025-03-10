from pathlib import Path
import logging
import ray
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from itertools import chain
from datetime import datetime

from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import apply_filtering_step
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote
class UnimodalVisionFilter:
    def __init__(self):
        pass

    def ray_unimodal_vision_filtering(self, tar_files: list[str | Path], log_folder: Path, config_path: Path):
        _ = Config(config_path)
        _ = MetricsStore()
        return unimodal_vision_filtering(tar_files, log_folder)


def unimodal_vision_filtering(tar_files: list[str | Path], log_folder: Path):
    logger = logging.getLogger("ray")

    # logger.info("Validating configuration")
    _validate_configuration()    
    
    # logger.info("Decoding webdataset")
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=False,
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

        # ------------------------------------------------------------------------ 
        # first step: filter by aspect ratio
        ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filtering_step(
            filter_name=_get_images_by_aspect_ratio_filter_mask,
            batch_id=batch_id,
            uids=batch[0],
            samples=batch[1],
            parameters = {
                "image_min_aspect_ratio": Config().unimodal.image_min_aspect_ratio,
                "image_max_aspect_ratio": Config().unimodal.image_max_aspect_ratio
            }
        )



        # filter_mask: list[bool]
        # filter_mask = _get_images_by_aspect_ratio_filter_mask(
        #     images,
        #     Config().unimodal.image_min_aspect_ratio,
        #     Config().unimodal.image_max_aspect_ratio
        # )
        # ok_uids, ok_images, uids_filtered, images_filtered = apply_filter_mask(
        #     uids, images, filter_mask,
        #     filter_name="_get_images_by_aspect_ratio_filter_mask",
        #     batch_id=batch_id
        # )


        # ------------------------------------------- 
        # second step: remove images containing text
        # TODO: implement this


        # ------------------------------------------- 
        # third step: image specificity filtering
        # TODO: implement this


        all_uids.append(ok_uids)


    # logger.info("Concatenating uids")
    all_uids = list(chain.from_iterable(all_uids))
    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if Config().infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)
    return all_uids


def _get_images_by_aspect_ratio_filter_mask(
        images: list[Image.Image],
        image_min_aspect_ratio: float,
        image_max_aspect_ratio: float
    ) -> list[bool]:
    """
    Filter the images by aspect ratio.
    """
    return [
        (image.width / image.height > image_min_aspect_ratio and image.width / image.height < image_max_aspect_ratio)
        for image in images
    ]

def _validate_configuration():
    config = Config()
    if config.unimodal.image_min_aspect_ratio < 0.0 or config.unimodal.image_min_aspect_ratio > 1.0:
        raise ValueError("The aspect ratio threshold must be between 0.0 and 1.0")
