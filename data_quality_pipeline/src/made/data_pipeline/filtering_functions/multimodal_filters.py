import math
from typing import List

import torch
import numpy as np
from PIL import Image

def filter_by_clip_similarity(
        captions: List[str],
        images: List[Image.Image],
        dfn_model,
        clip_processor,
        dfn_similarity_score_threshold: int,
        clip_caption_max_length: int
    ) -> List[bool]:
    """Filter by CLIP similarity scores using percentile threshold"""

    similarity_scores = []
    for img, txt in zip(images, captions):
        inputs = clip_processor(
            text=[txt],
            images=[img],
            return_tensors="pt",
            input_data_format="channels_last",
            padding=True,
            truncation=True,
            max_length=clip_caption_max_length
        ).to("cuda")
        outputs = dfn_model(**inputs)
        score = outputs.logits_per_image.item()
        similarity_scores.append(score)
    
    return [score > dfn_similarity_score_threshold for score in similarity_scores]

def filter_by_specificity(
        captions: List[str],
        images: List[Image.Image],
        meru_model,
        trs,
        tokenizer,
        img_ref,
        txt_ref,
        curvature,
        specificity_threshold: float,
        weight: float
    ) -> List[bool]:
    """Filter by specificity of the caption"""
    images = torch.stack([trs(im) for im in images]).to("cuda")

    with torch.no_grad():
        encoded_images = meru_model.encode_image(images)

    with torch.no_grad():
        tokenized_captions = tokenizer(captions, return_tensors="pt", padding="max_length", truncation=True, max_length=77)["input_ids"].to("cuda")
        encoded_captions = meru_model.encode_text(tokenized_captions)

    image_specificity_scores = _image_specificity(txt_ref=txt_ref, curv=curvature, image=encoded_images)
    text_specificity_scores = _text_specificity(img_ref=img_ref, curv=curvature, text=encoded_captions)

    image_text_specificity_score = text_specificity_scores * weight + image_specificity_scores * (1 - weight)

    return (image_text_specificity_score > specificity_threshold).tolist()


def _image_specificity(txt_ref: torch.Tensor, curv: float, image: torch.Tensor):
    txt_ref = txt_ref.to(image.device)
    ient = _entailment(txt_ref, image, curv)
    return ient.mean(dim=0)

def _text_specificity(img_ref: torch.Tensor, curv: float, text: torch.Tensor):    
    img_ref = img_ref.to(text.device)
    tent = _entailment(text, img_ref, curv)
    return tent.mean(dim=1)

@torch.no_grad()
def _entailment(x, y, curvature):
    x_space, x_time = _expm(x, curvature, time_keepdim=True)
    y_space, y_time = _expm(y, curvature, time_keepdim=True)

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
def _expm(v, curvature, time_keepdim=False):
    v, curvature = torch.tensor(v).float(), torch.tensor(curvature).float()
    x_space_temp = torch.sqrt(curvature) * torch.norm(v, dim=-1, keepdim=True)
    x_space = (
            torch.sinh(torch.clamp(x_space_temp, min=1e-8, max=math.asinh(2 ** 15))) * v / torch.clamp(x_space_temp, min=1e-8)
    )
    x_time = torch.sqrt(1 / curvature + torch.sum(x_space ** 2, dim=-1, keepdim=time_keepdim))
    return x_space, x_time