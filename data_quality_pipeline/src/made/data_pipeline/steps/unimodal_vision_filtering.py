from pathlib import Path

import ray
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from made.config import Config
from made.data_pipeline.metrics.metrics_decorators import get_time
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.metrics.utils import send_metrics_central_collector
from made.data_pipeline.utils import apply_filter_mask
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote
def ray_unimodal_vision_filtering(tar_files: list[str | Path]):
    results = unimodal_vision_filtering(tar_files)
    send_metrics_central_collector()
    return results

@get_time
def unimodal_vision_filtering(tar_files: list[str | Path]):

    _validate_configuration()    
    
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=False,
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
        images = batch[1]
        batch_id += 1
        sample_count += len(uids)

        # ------------------------------------------- 
        # first step: filter by aspect ratio
        filter_mask: list[bool]
        filter_mask = _get_images_by_aspect_ratio_filter_mask(
            images,
            Config().unimodal.image_min_aspect_ratio,
            Config().unimodal.image_max_aspect_ratio
        )
        ok_uids, ok_images, uids_filtered, images_filtered = apply_filter_mask(
            uids, images, filter_mask,
            filter_name="_get_images_by_aspect_ratio_filter_mask",
            batch_id=batch_id
        )


        # ------------------------------------------- 
        # second step: remove images containing text
        # TODO: implement this


        # ------------------------------------------- 
        # third step: image specificity filtering
        # TODO: implement this


        all_uids.append(ok_uids)


    all_uids = np.concatenate(all_uids)

    return all_uids

@get_time
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
