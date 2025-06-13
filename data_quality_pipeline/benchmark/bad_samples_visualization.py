import argparse
import glob
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")
from made.data_pipeline.utils import collect_tar_files
from collections import defaultdict
from utils import extract_samples_from_tar_files, create_samples_visualization

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
    parser.add_argument("-t", "--tar-files-path", type=str, required=False, default="/home/leobaro/workspace/labs/fair_spoke_8/data_quality_pipeline/benchmark/data")
    parser.add_argument("-n", "--num-samples", type=int, required=False, default=20)
    return parser.parse_args()

def main(args):
    tar_files = collect_tar_files(args.tar_files_path, recursive=False)

    bad_uids = glob.glob(os.path.join(args.folder, "baduids_*.txt"))

    filter_names = set([Path(f).stem.split("_")[1] for f in bad_uids])

    bad_uids_per_filter = defaultdict(list)
    for filter_name in filter_names:
        files = glob.glob(os.path.join(args.folder, f"baduids_{filter_name}*.txt"))
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                bad_uids_per_filter[filter_name].extend(f.read().splitlines())


    output_dir = Path(Path(args.folder) / "bad_uids_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filter_name, bad_uids_samples in bad_uids_per_filter.items():
        uids, images, captions = extract_samples_from_tar_files(bad_uids_samples, tar_files, get_images=False, get_captions=True, num_samples=5000)
        create_txt_file_list(uids, captions, Path(args.folder) / "bad_uids_visualizations" / f"{filter_name}.txt")
        
        uids, images, captions = extract_samples_from_tar_files(bad_uids_samples, tar_files, get_images=True, get_captions=True, num_samples=20)
        create_samples_visualization(uids, images, captions, f"Bad uids for {filter_name} filter", output_dir )


if __name__ == "__main__":
    main(cli())