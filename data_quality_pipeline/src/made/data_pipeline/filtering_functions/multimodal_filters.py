import torch
import numpy as np
from PIL import Image
from typing import List

def filter_by_clip_similarity(
        captions: List[str],
        images: List[Image.Image],
        dfn_model,
        clip_processor,
        dfn_percentile_to_drop: int,
        clip_caption_max_length: int
    ) -> List[bool]:
    """Filter by CLIP similarity scores using percentile threshold"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        ).to(device)
        outputs = dfn_model(**inputs)
        score = outputs.logits_per_image.item()
        similarity_scores.append(score)
    
    threshold = np.percentile(similarity_scores, dfn_percentile_to_drop)
    return [score > threshold for score in similarity_scores]