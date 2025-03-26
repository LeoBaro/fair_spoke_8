from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

import pytest
import PIL

def test_webdataset_reader_with_captions(tar_files):
    dataset = decode_webdataset(
        tar_files,
        get_images=False,
        get_captions=True,
        batch_size=16
    )
    batch = get_next_batch(iter(dataset))
    assert batch is not None
    assert len(batch) == 2

    assert len(batch[0]) == 16
    assert len(batch[1]) == 16
    assert isinstance(batch[0][0], str)
    assert isinstance(batch[1][0], str)

def test_webdataset_reader_with_images(tar_files):
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=False,
        batch_size=16
    )
    batch = get_next_batch(iter(dataset))
    assert batch is not None
    assert len(batch) == 2

    assert len(batch[0]) == 16
    assert len(batch[1]) == 16
    assert isinstance(batch[0][0], str)
    assert isinstance(batch[1][0], PIL.Image.Image)

def test_webdataset_reader_with_images_and_captions(tar_files):
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=16
    )
    batch = get_next_batch(iter(dataset))
    assert batch is not None
    assert len(batch) == 3

    assert len(batch[0]) == 16
    assert len(batch[1]) == 16
    assert isinstance(batch[0][0], str)
    assert isinstance(batch[1][0], PIL.Image.Image)
    assert isinstance(batch[2][0], str)
    

def test_webdataset_reader_with_valid_uids(tar_files):
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        valid_uids=[
            'd84cefb32975139db7f9cfa4e9f00fb0', 
            '7e69a66689f697d961c18f085af23d8c', 
            '4f42e5fd6e85a45b05714b85cd4d377d', 
            'd40f4f15a8c92ebebdb6a5785bf38b98'
        ]
    )   
    batch = get_next_batch(iter(dataset))
    assert batch is not None
    assert len(batch) == 3
    assert len(batch[0]) == 4
    assert len(batch[1]) == 4
    assert len(batch[2]) == 4
    