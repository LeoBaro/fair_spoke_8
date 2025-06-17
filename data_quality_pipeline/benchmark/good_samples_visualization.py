import os
import glob
import argparse
from pathlib import Path

from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from utils import create_samples_visualization

def get_good_uids(results_folder):
    good_uids_files = glob.glob(os.path.join(results_folder, "*.txt"))
    good_uids = []
    for file in good_uids_files:
        with open(file, "r", encoding="utf-8") as f:
            good_uids.extend(f.read().splitlines())
    return good_uids

def create_txt_file_list(uids: list[str], captions: list[str], output_file: str):
    print("UIDS length: ", len(uids))
    print("Captions length: ", len(captions))
    assert len(uids) == len(captions), "UIDS and Captions must have the same length"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for uid, caption in zip(uids, captions):
            f.write(f"{uid} {caption}\n")
    print(f"Visualization saved to {output_file}")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("-t", "--tar-files-path", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-i", "--number-of-samples-per-image", type=int, required=False, default=20)
    return parser.parse_args()

def main(args):
    output_dir_name = f"{Path(args.folder).name}_good_uids_visualizations"
    output_dir = Path(Path(args.folder).parent / output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = decode_webdataset(
        collect_tar_files(args.tar_files_path, recursive=False),
        get_images=True,
        get_captions=True,
        batch_size=args.number_of_samples_per_image
    )
    dataset = iter(dataset)

    count = 0
    while count < args.num_samples:
        try:
            batch_uids, batch_images, batch_captions = get_next_batch(dataset)
            output_path = output_dir / f"good_uids_{count}_{count + args.number_of_samples_per_image}"
            create_samples_visualization(batch_uids, batch_images, batch_captions, f"", output_path)
            count += args.number_of_samples_per_image
        except Exception as e:
            print("Exception!", e)
            break



if __name__ == "__main__":
    main(cli())