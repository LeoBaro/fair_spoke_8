import sys
import os

sys.path.append(os.path.abspath(".."))

import time
from data_quality_pipeline.src.made.data_pipeline.data.datacomp_handler import decode_webdataset, decode_webdataset_two_steps, get_next_batch
import PIL


def benchmark_webdataset(tar_files, iterations=10):
    configs = [
        {"get_images": False, "get_captions": True, "description": "Captions Only"},
        {"get_images": True, "get_captions": False, "description": "Images Only"},
        {"get_images": True, "get_captions": True, "description": "Images and Captions"},
        {"get_images": True, "get_captions": True, "valid_uids": [
            'd84cefb32975139db7f9cfa4e9f00fb0',
            '7e69a66689f697d961c18f085af23d8c',
            '4f42e5fd6e85a45b05714b85cd4d377d',
            'd40f4f15a8c92ebebdb6a5785bf38b98'
        ], "description": "Valid UIDs Filter"},
    ]

    for config in configs:
        total_time = 0

        for _ in range(iterations):
            dataset = decode_webdataset_two_steps(
                tar_files,
                get_images=config.get("get_images", False),
                get_captions=config.get("get_captions", False),
                batch_size=16,
                valid_uids=config.get("valid_uids", None)
            )

            start_time = time.perf_counter()
            batch = get_next_batch(iter(dataset))
            end_time = time.perf_counter()

            assert batch is not None
            total_time += (end_time - start_time)

        avg_time = total_time / iterations
        print(f"{config['description']} - Avg. Time: {avg_time:.6f} sec")


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # 10000 elements
    tar_files = [str(Path(__file__).parent / "data" / "00000000_reduced.tar")]
    benchmark_webdataset(tar_files)