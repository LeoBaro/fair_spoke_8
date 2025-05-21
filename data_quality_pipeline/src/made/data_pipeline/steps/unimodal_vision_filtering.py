from pathlib import Path
import logging  
import json
import ray
import easyocr
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from itertools import chain, compress
from datetime import datetime
from collections import defaultdict

from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import execute_filter, apply_filter_mask
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote(num_gpus=0.1)
class UnimodalVisionFilter:
    def __init__(self, config_path: Path):
        super().__init__()
        self.config = Config(config_path)
        self.reader = easyocr.Reader(['en'], gpu=True, user_network_directory=self.config.unimodal.text_detection_model_path)

    def execute(self, tar_files: list[str | Path], log_folder: Path, uids: list[str] = None):
        _ = MetricsStore()
        return unimodal_vision_filtering(
            self.reader,
            tar_files, 
            log_folder, 
            self.config, 
            uids
        )


def unimodal_vision_filtering(
        text_detection_model: easyocr.Reader,
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
        get_images=True,
        get_captions=False,
        batch_size=config.unimodal.batch_size,
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

        # ------------------------------------------------------------------------ 
        # first step: filter by aspect ratio
        good_uids = batch[0]
        good_images = batch[1]
        
        filter_fn_parameters = {
            "image_min_aspect_ratio": config.unimodal.image_min_aspect_ratio,
            "image_max_aspect_ratio": config.unimodal.image_max_aspect_ratio,
            "image_min_dimension": config.unimodal.image_min_dimension
        }
        aspect_ratio_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_images_by_aspect_ratio_filter_mask,
            captions=None,
            images=good_images,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            MetricsStore().add_filter_metric(
                "_get_images_by_aspect_ratio_filter_mask",
                batch_id,
                len(good_images),
                int(sum(aspect_ratio_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["image_min_aspect_ratio", "image_max_aspect_ratio", "image_min_dimension"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in aspect_ratio_filter_mask]))
            filtered_uids_by_filter["aspect_ratio"].extend(bad_uids)

        good_uids = list(compress(good_uids, [m for m in aspect_ratio_filter_mask]))
        good_images = list(compress(good_images, [m for m in aspect_ratio_filter_mask]))

        # ------------------------------------------- 
        # second step: remove images containing text
        filter_fn_parameters = {
            "model": text_detection_model,
            "text_thresh": config.unimodal.text_threshold,
            "mag_ratio": config.unimodal.text_detection_mag_ratio
        }
        text_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_images_by_text_filter_mask,
            captions=None,
            images=good_images,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            MetricsStore().add_filter_metric(
                "_get_images_by_text_filter_mask",
                batch_id,
                len(good_images),
                int(sum(text_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["model", "text_thresh", "mag_ratio"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in text_filter_mask]))
            filtered_uids_by_filter["text_detection"].extend(bad_uids)

        good_uids = list(compress(good_uids, [m for m in text_filter_mask]))
        good_images = list(compress(good_images, [m for m in text_filter_mask]))


        all_good_uids.append(good_uids)


    all_good_uids = list(chain.from_iterable(all_good_uids))

    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)

    if config.infrastructure.save_filtered_uids:
        filtered_uids_path = log_folder / "bad_uids_unimodal_vision_filtering.json"
        with open(filtered_uids_path, 'w', encoding="utf-8") as f:
            json.dump(filtered_uids_by_filter, f, indent=2)
        logger.info("Filtered UIDs saved to %s", filtered_uids_path)

    return all_good_uids


def _get_images_by_aspect_ratio_filter_mask(
        images: list[Image.Image],
        image_min_aspect_ratio: float,
        image_max_aspect_ratio: float,
        image_min_dimension: int
    ) -> list[bool]:
    """
    Filter the images by aspect ratio.
    """
    return [
        (
            (image.width / image.height > image_min_aspect_ratio )
            and (image.width / image.height < image_max_aspect_ratio)
            and (min(image.width, image.height) > image_min_dimension)
        )
        for image in images
    ]

def _get_images_by_text_filter_mask(
        images: list[Image.Image],
        model: easyocr.Reader,
        text_thresh: float,
        mag_ratio: float
    ) -> list[bool]:
    """
    Filter the images by text.
    """

    mask = [True] * len(images)
    img_array = [np.array(image) for image in images]

    for i, image in enumerate(img_array):
        text_results = model.readtext(
            image, 
            text_threshold = text_thresh,
            decoder = 'greedy',
            batch_size = 1,
            mag_ratio = mag_ratio,
        )

        if text_results:
            mask[i] = False

    return mask

def get_max_dimension(images: list[NDArray]) -> int:
    """
    Get the maximum dimension (width or height) across all images in a list.
    
    Args:
        images: List of numpy array images with shape (h, w, c)
        
    Returns:
        Maximum dimension value
    """
    max_width = max(img.shape[1] for img in images)  # Width is at index 1
    max_height = max(img.shape[0] for img in images)  # Height is at index 0
    
    return max(max_width, max_height)

def pad_images_to_min_size(
    images,
    target_width,
    target_height):
    """
    Add padding to images that have dimensions less than the specified width and height.
    
    Args:
        images: List of numpy array images
        target_width: Minimum width for the padded images
        target_height: Minimum height for the padded images
        
    Returns:
        List of padded numpy array images
    """
    padded_images = []
    
    for img in images:
        height, width = img.shape[:2]
        
        # Calculate padding dimensions
        pad_width = max(0, target_width - width)
        pad_height = max(0, target_height - height)
        
        if pad_width > 0 or pad_height > 0:
            # Calculate padding for each side
            top = pad_height // 2
            bottom = pad_height - top
            left = pad_width // 2
            right = pad_width - left
            
            # Get number of channels (handle both RGB and grayscale)
            if len(img.shape) == 3:
                # RGB image
                padded_img = np.pad(
                    img,
                    ((top, bottom), (left, right), (0, 0)),
                    mode='constant',
                    constant_values=255
                )
            else:
                # Grayscale image
                padded_img = np.pad(
                    img,
                    ((top, bottom), (left, right)),
                    mode='constant',
                    constant_values=255
                )
            
            padded_images.append(padded_img)
        else:
            # No padding needed
            padded_images.append(img)
            
    return padded_images

def _get_images_by_text_filter_mask_batched(
        images: list[Image.Image],
        model: easyocr.Reader,
        text_thresh: float,
        mag_ratio: float
    ) -> list[bool]:
    """
    Filter the images by text.
    """

    mask = [True] * len(images)
    img_array = [np.array(image) for image in images]
    max_dimension = get_max_dimension(img_array)
    img_array = pad_images_to_min_size(img_array, max_dimension, max_dimension)

    text_results = model.readtext_batched(
        img_array,
        # n_width=256*3,
        # n_height=256*3,
        decoder = 'greedy',
        mag_ratio=mag_ratio,
        batch_size = len(img_array),
        paragraph = True,
        text_threshold=text_thresh
        )

    mask = [len(result) > 0 for result in text_results]

    return mask



def _validate_configuration(config: Config):
    if config.unimodal.image_min_aspect_ratio < 0.0 or config.unimodal.image_min_aspect_ratio > 1.0:
        raise ValueError("The aspect ratio threshold must be between 0.0 and 1.0")
