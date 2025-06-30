import argparse
from pathlib import Path
from collections import Counter

from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

def count_duplicates(uids):
    counter = Counter(uids)
    total = len(uids)
    unique = len(counter)
    duplicates = sum(count - 1 for count in counter.values() if count > 1)
    return total, unique, duplicates

def load_full_dataset(tar_files):
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=10_000_000
    )
    dataset = iter(dataset)
    return get_next_batch(dataset)

def main():
    args = cli()
    tar_files = collect_tar_files(args.web_dataset_folder, recursive=True)

    if not tar_files:
        raise FileNotFoundError("No .tar files found in input folder.")


    print(f"Loading dataset from {tar_files}")
    uids, _, _ = load_full_dataset(tar_files)
    total, unique, duplicates = count_duplicates(uids)

    print(f"Total samples: {total}")
    print(f"Unique samples: {unique}")
    print(f"Duplicate samples: {duplicates}")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--web-dataset-folder", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()
