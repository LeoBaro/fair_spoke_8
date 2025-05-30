from typing import Callable, Any
import time
from data_quality_pipeline.src.made.data_pipeline.metrics.metrics_store import MetricsStore
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import logging

class FilteringBlock(ABC):

    def __init__(self):
        self.logger = logging.getLogger("ray")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            raise ValueError("Filtering block is not supported on CPU")

    @abstractmethod
    def execute(self, tar_files: list[str | Path], log_folder: Path, uids: list[str] = None):
        pass
    

def execute_filter(
        filter_name: Callable,
        captions: list[str],
        images: list[Image.Image],
        parameters: dict[str, Any]
    ) -> list[bool]:
    """
    Apply a filter to the samples and return a boolean mask
    """
    start_time = time.time()
    if captions is not None and images is not None:
        boolean_mask = filter_name(captions, images, **parameters)
    elif captions is not None:
        boolean_mask = filter_name(captions, **parameters)
    elif images is not None:
        boolean_mask = filter_name(images, **parameters)
    else:
        raise ValueError("No samples to filter")
    elapsed_time = time.time() - start_time
    return  boolean_mask, elapsed_time

def apply_filter_mask(
        uids: list[str],
        mask: list[bool],
    ) -> tuple[list[str], list[str]]:
    """
    Apply a filter mask to items and data, returning both kept and filtered items
    
    Args:
        items: List of identifiers (e.g., UIDs)
        mask: Boolean mask for filtering
    Returns:
        Tuple of (kept_items, filtered_items)
    """
    kept_uids = []
    filtered_uids = []

    for item, m in zip(uids, mask):
        if m:  # Keep this item
            kept_uids.append(item)
        else:  # Filter out this item
            filtered_uids.append(item)

    return kept_uids, filtered_uids
