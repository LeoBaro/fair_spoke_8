import ray
from spoke8leonardo.data_pipeline.config import Config

@ray.remote
def filter_caption_length(batch):
    unimodal_config = Config().get_unimodal_config()
    return dataset.filter(
        lambda sample: CAPTION_MIN_LENGTH <= len(sample["caption"].split()) <= CAPTION_MAX_LENGTH
    )