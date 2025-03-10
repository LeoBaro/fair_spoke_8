import io
import json
from pathlib import Path

from PIL import Image
import imageio.v2 as imageio
import webdataset as wds

def decode_image(value):
    return Image.fromarray(imageio.imread(io.BytesIO(value)))

def decode_uid(value):
    return json.loads(value.decode("utf-8")).get("uid", "unknown")  

def decode_caption(value):
    return value.decode("utf-8").strip()

def decode_webdataset(
        tar_files: list[str | Path],
        batch_size: int = 16,
        get_images: bool = True,
        get_captions: bool = True,
) -> wds.compat.WebDataset:

    if len(tar_files) == 0:
        raise ValueError("The tar files list is empty")
    
    # always decode uid    
    decoders = [wds.handle_extension(".json", decode_uid)]
    keys = ["json"]

    # decode image if needed
    if get_images:
        decoders.append(wds.handle_extension(".jpg", decode_image))
        keys.append("jpg")

    # decode caption if needed
    if get_captions:
        decoders.append(wds.handle_extension(".txt", decode_caption))
        keys.append("txt")

    return (
        wds.WebDataset(tar_files, shardshuffle=False)
        .decode(
            *decoders
        )
        .to_tuple(*keys)
        .batched(batch_size)
    )

def get_next_batch(dataset_iter: iter):
    try:
        return next(dataset_iter)
    except StopIteration:
        return None