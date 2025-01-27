import ray
from PIL import Image
from spoke8leonardo.data_pipeline.config import Config

@ray.remote
def filter_image_aspect_ratio(batch):
    unimodal_config = Config().get_unimodal_config()
    filtered = []
    for sample in batch:
        image = Image.open(sample["image"])
        width, height = image.size
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio <= unimodal_config["aspect_ratio_threshold"]:
            filtered.append(sample)
    return filtered