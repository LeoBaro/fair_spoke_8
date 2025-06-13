import time
import json
import logging  
from pathlib import Path
from itertools import compress
from collections import defaultdict

import ray
import easyocr
import numpy as np
from PIL import Image
from numpy.typing import NDArray

from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import execute_filter, FilteringBlock, FilteringResult
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

@ray.remote(num_gpus=0.1, max_concurrency=1)
class UnimodalVisionFilter(FilteringBlock):

    def __init__(self, config_path: Path, log_folder: Path, webdataset_output_folder: Path):
        super().__init__(config_path, log_folder, webdataset_output_folder)
        self.logger.info("Initializing UnimodalVisionFilter on %s", self.device)
        self.logger.info("Number of workers: %s", self.config.infrastructure.num_workers)
        self.logger.info("Batch size: %s", self.config.infrastructure.batch_size)
        self.logger.info("Log folder: %s", self.log_folder)
        self.logger.info("Output folder: %s", self.filtering_result.output_folder)
        self.reader = easyocr.Reader(['en'], gpu=True, user_network_directory=self.config.unimodal_vision.text_detection_model_path)

    def execute(self, tar_files: list[str | Path]):
        return unimodal_vision_filtering(
            tar_files, 
            self.reader,
            self.config,
            self.metrics_store,
            self.filtering_result
        )


def unimodal_vision_filtering(
        tar_files: list[str | Path], 
        text_detection_model: easyocr.Reader,
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
        batch_size=config.unimodal_vision.batch_size
    )   
    
    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    logger.info("Starting unimodal vision filtering")
    start_time = time.time()

    while True:
        batch_start_time = time.time()

        batch = get_next_batch(dataset_iter)
        if batch is None:
            break

        batch_id += 1
        sample_count += len(batch[0])

        # ------------------------------------------------------------------------ 
        # first step: filter by aspect ratio
        good_uids = batch[0]
        good_images = batch[1]
        good_captions = batch[2]

        filter_fn_parameters = {
            "image_min_aspect_ratio": config.unimodal_vision.image_min_aspect_ratio,
            "image_max_aspect_ratio": config.unimodal_vision.image_max_aspect_ratio,
            "image_min_dimension": config.unimodal_vision.image_min_dimension
        }
        aspect_ratio_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_images_by_aspect_ratio_filter_mask,
            captions=None,
            images=good_images,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            metrics_store.add_filter_metric(
                "_get_images_by_aspect_ratio_filter_mask",
                len(good_images),
                int(sum(aspect_ratio_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["image_min_aspect_ratio", "image_max_aspect_ratio", "image_min_dimension"]
            )
        if config.infrastructure.save_bad_uids:
            metrics_store.dump_bad_uids("aspect_ratio", list(compress(good_uids, [not m for m in aspect_ratio_filter_mask])))

        good_uids = list(compress(good_uids, [m for m in aspect_ratio_filter_mask]))
        good_images = list(compress(good_images, [m for m in aspect_ratio_filter_mask]))
        good_captions = list(compress(good_captions, [m for m in aspect_ratio_filter_mask]))

        # ------------------------------------------- 
        # second step: remove images containing text
        filter_fn_parameters = {
            "model": text_detection_model,
            "text_thresh": config.unimodal_vision.text_threshold,
            "mag_ratio": config.unimodal_vision.text_detection_mag_ratio
        }
        text_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_images_by_text_filter_mask,
            captions=None,
            images=good_images,
            parameters = filter_fn_parameters
        )
        if config.infrastructure.enable_metrics:
            metrics_store.add_filter_metric(
                "_get_images_by_text_filter_mask",
                len(good_images),
                int(sum(text_filter_mask)),
                elapsed_time,
                filter_fn_parameters,
                ["model", "text_thresh", "mag_ratio"]
            )
        if config.infrastructure.save_bad_uids:
            metrics_store.dump_bad_uids("text_detection", list(compress(good_uids, [not m for m in text_filter_mask])))

        filtering_result.add_samples(
            list(compress(good_uids, [m for m in text_filter_mask])), 
            list(compress(good_captions, [m for m in text_filter_mask])), 
            list(compress(good_images, [m for m in text_filter_mask]))
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
    if config.unimodal_vision.image_min_aspect_ratio < 0.0 or config.unimodal_vision.image_min_aspect_ratio > 1.0:
        raise ValueError("The aspect ratio threshold must be between 0.0 and 1.0")
