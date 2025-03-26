import ray
import torch
import math
import os
import logging
import numpy as np
from pathlib import Path
from itertools import chain
from datetime import datetime
from PIL import Image

from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.steps.base import apply_filtering_step
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch


@ray.remote
class SpecificityFilter:
    def __init__(self, config_path: Path):
        self.config = Config(config_path)
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.pt")
        ref = torch.load(ref_path)
        self.img_ref, self.txt_ref = ref["img"], ref["txt"]

    def ray_specificity_filtering(self, tar_files: list[str | Path], log_folder: Path):
        _ = MetricsStore()  # Metrics tracking if enabled
        return specificity_filtering(tar_files, log_folder, self.config, self.img_ref, self.txt_ref)



def specificity_filtering(tar_files: list[str | Path], log_folder: Path, config: Config, img_ref: torch.Tensor, txt_ref: torch.Tensor):
    logger = logging.getLogger("ray")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_ref.to(device)
    txt_ref.to(device)

    # Validate configuration
    _validate_configuration(config)

    # Decode dataset
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=config.unimodal.batch_size
    )

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

        # Convert batch images to tensors and move them to device
        batch_images = [torch.tensor(np.array(img)).to(device) for img in batch[1]]
        captions = batch[2]
        
        # Apply specificity filtering
        ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filtering_step(
            filter_name=_get_images_by_specificity_filter_mask,
            batch_id=batch_id,
            uids=batch[0],
            samples=batch_images,
            parameters={
                "specificity_threshold": config.multimodal.specificity_threshold,
                "curvature": config.multimodal.curvature,
            }
        )

        all_uids.append(ok_uids)

    all_uids = list(chain.from_iterable(all_uids))
    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)
    
    return all_uids


def _get_images_by_specificity_filter_mask(
        images: list[torch.Tensor],
        specificity_threshold: float,
        curvature: float
    ) -> list[bool]:
    """
    Filter images based on specificity.
    """
    return [specificity(image=image, curv=curvature) > specificity_threshold for image in images]


def _validate_configuration(config: Config):
    if config.unimodal.specificity_threshold < 0.0 or config.unimodal.specificity_threshold > 1.0:
        raise ValueError("The specificity threshold must be between 0.0 and 1.0")
    if config.unimodal.curvature <= 0.0:
        raise ValueError("Curvature must be a positive value")


def specificity(image, text, curv: float):

    if image is not None:
        txt_ref = txt_ref.to(image.device)  # Ensure txt_ref is on the same device as the image
        ient = entailment(txt_ref, image, curv)
        return ient.mean(dim=0)
    else:
        img_ref = img_ref.to(text.device)  # Ensure img_ref is on the same device as the text
        tent = entailment(text, img_ref, curv)
        return tent.mean(dim=1)


def entailment(x, y, curvature):
    x_space, x_time = expm(x, curvature, time_keepdim=True)
    y_space, y_time = expm(y, curvature, time_keepdim=True)

    K = 0.1
    x_euc_norm = torch.norm(x_space, dim=-1, keepdim=True)
    denominator = torch.sqrt(curvature) * x_euc_norm + 1e-8
    aperture_x = torch.arcsin(torch.clamp(2 * K / denominator, -1 + 1e-8, 1 - 1e-8))

    xy_inner = x_space @ y_space.T - x_time * y_time.T
    denominator = x_euc_norm * torch.sqrt(torch.clamp((curvature * xy_inner) ** 2 - 1, min=1e-8)) + 1e-8
    numerator = y_time.T + x_time * curvature * xy_inner
    exterior_xy = torch.arccos(torch.clamp(numerator / denominator, -1.0 + 1e-8, 1.0 - 1e-8))

    return exterior_xy - aperture_x


def expm(v, curvature, time_keepdim=False):
    v, curvature = v.float(), curvature.float()
    x_space_temp = torch.sqrt(curvature) * torch.norm(v, dim=-1, keepdim=True)
    x_space = (
        torch.sinh(torch.clamp(x_space_temp, min=1e-8, max=math.asinh(2**15))) * v / torch.clamp(x_space_temp, min=1e-8)
    )
    x_time = torch.sqrt(1 / curvature + torch.sum(x_space**2, dim=-1, keepdim=time_keepdim))
    return x_space, x_time

