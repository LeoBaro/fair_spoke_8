import numpy as np
import easyocr
from PIL import Image
from typing import List

def filter_images_by_aspect_ratio(
        images: List[Image.Image],
        image_min_aspect_ratio: float,
        image_max_aspect_ratio: float,
        image_min_dimension: int
    ) -> List[bool]:
    """Filter images by aspect ratio and minimum dimension"""
    return [
        (image_min_aspect_ratio < image.width / image.height < image_max_aspect_ratio)
        and (min(image.width, image.height) > image_min_dimension)
        for image in images
    ]

def filter_images_by_text_detection(
        images: List[Image.Image],
        model: easyocr.Reader,
        text_thresh: float,
        mag_ratio: float
    ) -> List[bool]:
    """Filter out images containing text"""
    mask = []
    for image in images:
        img_array = np.array(image)
        text_results = model.readtext(
            img_array,
            text_threshold=text_thresh,
            decoder='greedy',
            batch_size=1,
            mag_ratio=mag_ratio,
        )
        mask.append(len(text_results) == 0)  # Keep images with no text
    return mask

