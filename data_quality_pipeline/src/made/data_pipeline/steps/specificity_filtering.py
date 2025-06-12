import ray
import torch
import torch.nn as nn
import math
import os
import logging
import numpy as np
from pathlib import Path
from itertools import chain, compress
from datetime import datetime
from PIL import Image
from collections import defaultdict

from data_quality_pipeline.src.made.config import Config
from data_quality_pipeline.src.made.data_pipeline.metrics.metrics_store import MetricsStore
from data_quality_pipeline.src.made.data_pipeline.steps.base import execute_filter, FilteringBlock
from data_quality_pipeline.src.made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from data_quality_pipeline.src.made.data_pipeline.model_hype import model_init

@ray.remote(num_gpus=1)
class SpecificityFilter(FilteringBlock):
    def __init__(self, config_path: Path):
        self.config = Config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            raise ValueError("Specificity filtering is not supported on cpu")

        # Load reference embeddings
        ref_path = "/davinci-1/work/fdimatteo/hype_weights/reference.pt"
        ref = torch.load(ref_path)
        self.img_ref = ref["img"].to(self.device)
        self.txt_ref = ref["txt"].to(self.device)

        # Load model
        self.model_weights = "/archive/SSD/home/fdimatteo/Progetti/fair_spoke_8/meru/hype/ckpt.pt"
        self.model, self.trs = model_init(pretrained=self.model_weights)
        self.model = self.model.to(self.device).eval()

    def execute(self, tar_files: list[str | Path], log_folder: Path, get_specificities = False):
        _ = MetricsStore()  # Metrics tracking if enabled


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
        get_specificities = True,
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

    # logger.info("Iterating over dataset")
    all_good_uids = []
    filtered_uids_by_filter = defaultdict(list)

    sample_count = 0
    batch_id = 0
    dataset_iter = iter(dataset)

    while batch_id<3:
        batch = get_next_batch(dataset_iter)
        if batch is None:
            break

        batch_id += 1
        sample_count += len(batch[1])

        # Convert batch images to tensors, move them to device and encode them
        images_tensors = torch.stack([trs(im) for im in batch[1]])
        images_tensors = images_tensors.to(device)

        with torch.no_grad():
            images_embeddings = model.encode_image(images_tensors)

        # Apply specificity filtering
        good_uids = batch[0]
        good_images = images_embeddings
        specificity_parameters = {
            "specificity_threshold": config.specificity.specificity_threshold,
            "curvature": torch.tensor(config.specificity.curvature, dtype=torch.float32, device=device),
            "img_ref": img_ref,
            "txt_ref": txt_ref,
            "get_specificities": get_specificities,
            }

        specificity_filter_mask, elapsed_time = execute_filter(
            filter_name=_get_images_specificity_filter_mask,
            captions=None,
            images=good_images,
            parameters = specificity_parameters
        )

        if config.infrastructure.enable_metrics:
            MetricsStore().add_filter_metric(
                "_get_specificity_filter_mask",
                batch_id,
                len(batch[0]),
                int(sum(specificity_filter_mask)),
                elapsed_time,
                specificity_parameters,
                ["specificity_threshold"]
            )
        if config.infrastructure.save_filtered_uids:
            bad_uids = list(compress(good_uids, [not m for m in specificity_filter_mask]))
            filtered_uids_by_filter["specificity"].extend(bad_uids)

        good_uids = list(compress(good_uids, [m for m in specificity_filter_mask]))
        # good_images = list(compress(good_images, [m for m in specificity_filter_mask]))


        all_good_uids.append(good_uids)


    # logger.info("Concatenating uids")
    all_good_uids = list(chain.from_iterable(all_good_uids))

    logger.info(f"[{datetime.now()}] Total samples processed: %s", sample_count)

    if config.infrastructure.enable_metrics:
        MetricsStore().save_to_file(log_folder)

    if config.infrastructure.save_filtered_uids:
        filtered_uids_path = log_folder / "bad_uids_multimodal_filtering.json"
        with open(filtered_uids_path, 'w', encoding="utf-8") as f:
            json.dump(filtered_uids_by_filter, f, indent=2)
        logger.info("Filtered UIDs saved to %s", filtered_uids_path)

    return all_good_uids


def _get_images_specificity_filter_mask(
        images: torch.Tensor,
        specificity_threshold: float,
        curvature: float,
        img_ref: torch.Tensor,
        txt_ref: torch.Tensor,
        get_specificities = False,
        ) -> list[bool] | tuple[list[bool], torch.Tensor]:
        """
        Filter images based on specificity.
        """
        specifities = specificity(image=images, curv=curvature, img_ref=img_ref, txt_ref=txt_ref)
        print(specifities)

        if get_specificities:
            return ((specifities > specificity_threshold).tolist(), specifities)

        return (specifities > specificity_threshold).tolist()


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

