import ray
import torch
import torch.nn as nn
import math
import logging
import numpy as np
from pathlib import Path
from itertools import chain
from datetime import datetime
from PIL import Image
from tokenizer import tokenize

from data_quality_pipeline.src.made.config import Config
from data_quality_pipeline.src.made.data_pipeline.metrics.metrics_store import MetricsStore
from data_quality_pipeline.src.made.data_pipeline.steps.base import apply_filtering_step, FilteringBlock
from data_quality_pipeline.src.made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from data_quality_pipeline.src.made.data_pipeline.model_hype import model_init

@ray.remote(num_gpus=1)
class SpecificityFilter(FilteringBlock):
    def __init__(self, config_path: Path):
        self.config = Config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load reference embeddings
        ref_path = "/davinci-1/work/fdimatteo/hype_weights/reference.pt"
        ref = torch.load(ref_path)
        self.img_ref = ref["img"].to(self.device)
        self.txt_ref = ref["txt"].to(self.device)

        # Load model
        self.model_weights = "/archive/SSD/home/fdimatteo/Progetti/fair_spoke_8/meru/hype/ckpt.pt"
        self.model, self.trs = model_init(pretrained=self.model_weights)
        self.model = self.model.to(self.device).eval()

    def execute(self, tar_files: list[str | Path], log_folder: Path, hype_score = False, get_specificities = False):
        _ = MetricsStore()  # Metrics tracking if enabled


        print("CUDA available inside Ray actor:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA current device:", torch.cuda.current_device())

        return specificity_filtering(
            tar_files,
            log_folder,
            self.config,
            self.model,
            self.device,
            self.trs,
            self.img_ref,
            self.txt_ref,
            get_specificities=False,
        )


def specificity_filtering(
        tar_files: list[str | Path],
        log_folder: Path,
        config: Config,
        model: nn.Module,
        device: torch.device,
        trs,
        img_ref: torch.Tensor,
        txt_ref: torch.Tensor,
        get_specificities = False,
):
    logger = logging.getLogger("ray")

    # Validate configuration
    _validate_configuration(config)

    # Decode dataset
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=config.specificity.batch_size
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

        # Convert batch images to tensors, move them to device and encode them
        batch_images = torch.stack([trs(im).to(device) for im in batch[0]])

        with torch.no_grad():
            images_features = model.encode_image(batch_images)
            text_features = model.encode_text(batch_text)

        # Apply specificity/hype filtering

        ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filtering_step(
            filter_name=_get_images_by_hype_filter_mask,
            batch_id=batch_id,
            uids=batch[1],
            samples=images_features,
            apply_filters=config.infrastructure.apply_filters,
            parameters={
                "specificity_threshold": config.specificity.hype_threshold,
                "curvature": torch.tensor(config.specificity.curvature, dtype=torch.float32, device=device),
                "img_ref": img_ref,
                "txt_ref": txt_ref,
                "get_specificities": get_specificities,
            }
        )

        all_uids.append(ok_uids)

    all_uids = list(chain.from_iterable(all_uids))
    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)

    return all_uids


def _get_images_by_hype_filter_mask(
        images: torch.Tensor,
        captions: torch.Tensor,
        hype_threshold: float,
        curvature: float,
        get_specificities=False,
        ) -> list[bool] | tuple[list[bool], torch.Tensor]:
        """
        Filter images based on hype score.
        """
        specifities = specificity(image=images, curv=curvature, img_ref=img_ref, txt_ref=txt_ref)
        meru_sim = similarity(images, captions, curv=curvature)
        hype = specifities  + meru_sim

        if get_specificities:
            return ((specifities > specificity_threshold).tolist(), specifities)

        return (hype > hype_threshold).tolist()


def _validate_configuration(config: Config):
    ##if config.specificity_threshold < 0.0 or config.specificity_threshold > 1.0:
    # raise ValueError("The specificity threshold must be between 0.0 and 1.0")
    if config.specificity.curvature <= 0.0:
        raise ValueError("Curvature must be a positive value")


def specificity(img_ref: torch.Tensor, txt_ref: torch.Tensor, curv = None, image=None, text=None):
    assert (image is not None) ^ (text is not None), "Either image or text must be provided but not both"
    assert curv is not None, "Curvature must be provided"


    if image is not None:
        txt_ref = txt_ref.to(image.device)
        ient = entailment(txt_ref, image, curv)
        return ient.mean(dim=0)
    else:
        img_ref = img_ref.to(text.device)
        tent = entailment(text, img_ref, curv)
        return tent.mean(dim=1)

@torch.no_grad()
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

@torch.cuda.amp.autocast(enabled=False)
def expm(v, curvature, time_keepdim=False):
    v, curvature = v.float(), curvature.float()
    x_space_temp = torch.sqrt(curvature) * torch.norm(v, dim=-1, keepdim=True)
    x_space = (
            torch.sinh(torch.clamp(x_space_temp, min=1e-8, max=math.asinh(2 ** 15))) * v / torch.clamp(x_space_temp,
                                                                                                       min=1e-8)
    )
    x_time = torch.sqrt(1 / curvature + torch.sum(x_space ** 2, dim=-1, keepdim=time_keepdim))
    return x_space, x_time

@torch.cuda.amp.autocast(enabled=False)
def similarity(x, y, curvature):
    x, y = x.float(), y.float()
    curvature = curvature.float()
    x_space, x_time = expm(x, curvature, time_keepdim=True)
    y_space, y_time = expm(y, curvature, time_keepdim=True)
    xy_inner = x_space @ y_space.T - x_time * y_time.T
    lorentzian_distance = torch.rsqrt(curvature) * torch.acosh(torch.clamp(-curvature * xy_inner, min=1e-8))
    return -lorentzian_distance