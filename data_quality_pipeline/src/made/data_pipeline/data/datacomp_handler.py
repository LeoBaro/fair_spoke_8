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
        valid_uids: list[str] = None
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

    if valid_uids:
        # This is extremely slow
        def filter_by_uid(sample):
            uid = sample.get("json", "unknown")
            return uid in valid_uids if valid_uids else True

        return (
            wds.WebDataset(tar_files, shardshuffle=False)
            .decode(
                *decoders
            )
            .select(filter_by_uid)
            .to_tuple(*keys)
            .batched(batch_size)
        )
    return (
        wds.WebDataset(tar_files, shardshuffle=False)
        .decode(
            *decoders
        )
        .to_tuple(*keys)
        .batched(batch_size)
    )


def decode_webdataset_two_steps(
        tar_files: list[str | Path],
        batch_size: int = 16,
        get_images: bool = True,
        get_captions: bool = True,
        valid_uids: list[str] = None
) -> wds.compat.WebDataset:
    if len(tar_files) == 0:
        raise ValueError("The tar files list is empty")

    # always decode uid
    uids_decoder = [wds.handle_extension(".json", decode_uid)]
    keys = ["json"]

    decoders = []

    # decode image if needed
    if get_images:
        decoders.append(wds.handle_extension(".jpg", decode_image))
        keys.append("jpg")

    # decode caption if needed
    if get_captions:
        decoders.append(wds.handle_extension(".txt", decode_caption))
        keys.append("txt")

    if valid_uids:
        def filter_by_uid(sample):
            uid = sample.get("json", "unknown")
            return uid in valid_uids

        return (
            wds.WebDataset(tar_files, shardshuffle=False)
            .decode(*uids_decoder, only = ["json"])
            .select(filter_by_uid)
            .decode(*decoders, only = ["txt","jpg"])
            .to_tuple(*keys)
            .batched(batch_size)
        )
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