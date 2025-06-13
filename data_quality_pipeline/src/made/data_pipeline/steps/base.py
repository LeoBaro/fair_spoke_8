import io
import json
import time
import logging
import tarfile
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Callable, Any

import ray
import torch
from PIL import Image

from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore

class FilteringResult:

    def __init__(self, output_folder: Path, dump_every_n_samples: int):
        self.logger = logging.getLogger("ray")
        self.output_folder = Path(output_folder)
        self.dump_every_n_samples = dump_every_n_samples
        self.worker_id = ray.get_runtime_context().get_worker_id() if ray.is_initialized() else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.uids = []
        self.captions = []
        self.images = []
        self.dump_counter = 0
        self.produced_tar_files = []
        self.produced_uids_files = []

    def add_samples(self, uids: list[str], captions: list[str], images: list[Image.Image]):
        assert len(uids) == len(captions) == len(images), "All input lists must be of equal length"
        self.uids.extend(uids)
        self.captions.extend(captions)
        self.images.extend(images)

    def reset_data(self):
        self.uids = []
        self.captions = []
        self.images = []

    def dump_to_disk(self, force: bool = False):
        if len(self.uids) > self.dump_every_n_samples or force:
            start_time = time.time()
            number_of_samples = len(self.uids)
            self.dump_tar()
            self.dump_uids()
            self.reset_data()
            self.dump_counter += 1
            self.logger.info("Dumped %d samples to webdataset took %0.2f seconds. Number of tar files produced: %d", number_of_samples, time.time() - start_time, self.dump_counter)
            return True
        return False

    def dump_uids(self):
        uids_path = self.output_folder / f"{self.worker_id}_uids.txt"
        with open(uids_path, "a", encoding="utf-8") as f:
            for uid in self.uids:
                f.write(uid + "\n")
        self.produced_uids_files.append(uids_path)
    
    def dump_tar(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        tar_path = self.output_folder / f"{self.worker_id}_{self.dump_counter:08d}.tar"
        with tarfile.open(tar_path, "w") as tar:
            for idx, (uid, caption, image) in enumerate(zip(self.uids, self.captions, self.images)):
                base_name = f"{idx:08d}"

                txt_data = caption.encode("utf-8")
                txt_info = tarfile.TarInfo(name=f"{base_name}.txt")
                txt_info.size = len(txt_data)
                tar.addfile(txt_info, io.BytesIO(txt_data))

                img_buffer = io.BytesIO()
                image.save(img_buffer, format="JPEG")
                img_data = img_buffer.getvalue()
                img_info = tarfile.TarInfo(name=f"{base_name}.jpg")
                img_info.size = len(img_data)
                tar.addfile(img_info, io.BytesIO(img_data))

                json_obj = {"uid": uid}
                json_data = json.dumps(json_obj).encode("utf-8")
                json_info = tarfile.TarInfo(name=f"{base_name}.json")
                json_info.size = len(json_data)
                tar.addfile(json_info, io.BytesIO(json_data))
        self.produced_tar_files.append(tar_path)


class FilteringBlock(ABC):

    def __init__(self, config_path: Path, log_folder: Path, output_folder: Path):
        self.logger = logging.getLogger("ray")
        self.device = "cuda" 
        if not torch.cuda.is_available():        
            raise ValueError("Filtering block is not supported on CPU")
        self.logger.info("CUDA device count: %s", torch.cuda.device_count())
        self.logger.info("CUDA current device: %s", torch.cuda.current_device())

        self.config = Config(config_path)
        self.log_folder = log_folder
        self.filtering_result = FilteringResult(output_folder, self.config.infrastructure.dump_tar_every_n_samples)
        self.metrics_store = MetricsStore(self.log_folder)
        self.execution_counter = 0

    @abstractmethod
    def execute(self, tar_files: list[str | Path]):
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

# def apply_filter_mask(
#         uids: list[str],
#         mask: list[bool],
#     ) -> tuple[list[str], list[str]]:
#     """
#     Apply a filter mask to items and data, returning both kept and filtered items
    
#     Args:
#         items: List of identifiers (e.g., UIDs)
#         mask: Boolean mask for filtering
#     Returns:
#         Tuple of (kept_items, filtered_items)
#     """
#     kept_uids = []
#     filtered_uids = []

#     for item, m in zip(uids, mask):
#         if m:  # Keep this item
#             kept_uids.append(item)
#         else:  # Filter out this item
#             filtered_uids.append(item)

#     return kept_uids, filtered_uids
