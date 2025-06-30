import argparse
from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
import matplotlib.pyplot as plt
import tarfile
from pathlib import Path
import io
import json
 


def dump_tar(output_folder, tar_file_name, uids, captions, images):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    tar_path = output_folder / f"{tar_file_name}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for idx, (uid, caption, image) in enumerate(zip(uids, captions, images)):
            base_name = f"{idx:08d}"

            txt_data = caption.encode("utf-8")
            txt_info = tarfile.TarInfo(name=f"{base_name}.txt")
            txt_info.size = len(txt_data)
            tar.addfile(txt_info, io.BytesIO(txt_data))

            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_data = img_buffer.getvalue()
            img_info = tarfile.TarInfo(name=f"{base_name}.jpg")
            img_info.size = len(img_data)
            tar.addfile(img_info, io.BytesIO(img_data))

            json_obj = {"uid": uid}
            json_data = json.dumps(json_obj).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{base_name}.json")
            json_info.size = len(json_data)
            tar.addfile(json_info, io.BytesIO(json_data))


def deduplicate_dataset(dataset):
    """Remove duplicates by UID while preserving order."""
    seen_uids = set()
    dedup_uids, dedup_images, dedup_captions = [], [], []

    for uid, img, cap in zip(*dataset):
        if uid not in seen_uids:
            seen_uids.add(uid)
            dedup_uids.append(uid)
            dedup_images.append(img)
            dedup_captions.append(cap)

    return dedup_uids, dedup_images, dedup_captions

def load_full_dataset(tar_files):
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=10_000_000  # large batch to load everything
    )
    dataset = iter(dataset)
    return get_next_batch(dataset)

def main():
    args = cli()
    tar_files = collect_tar_files(args.web_dataset_folder, recursive=True)

    if not tar_files:
        raise FileNotFoundError("No .tar files found in input folder.")


    # Load and deduplicate
    print(f"Loading dataset from {tar_files}")
    uids, images, captions = load_full_dataset(tar_files)
    print(f"Loaded {len(uids)} samples")

    print("Deduplicating dataset...")
    uids, images, captions = deduplicate_dataset((uids, images, captions))
    print(f"{len(uids)} unique samples after deduplication")

    # Write to new tar
    dump_tar(args.output_folder, Path(tar_files[0]).stem, uids, captions, images)
    print(f"Saved deduplicated dataset to {args.output_folder}")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--web-dataset-folder", type=str, required=True)
    parser.add_argument("-o", "--output-folder", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()
