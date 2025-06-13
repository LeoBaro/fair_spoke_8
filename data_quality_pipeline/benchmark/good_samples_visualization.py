import argparse
import glob
import os
import random
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")
from made.data_pipeline.utils import collect_tar_files
from collections import defaultdict
from utils import extract_samples_from_tar_files, create_samples_visualization

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


    output_dir = Path(Path(args.folder) / "good_uids_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    good_uids = get_good_uids(args.folder)
    random.shuffle(good_uids)
    good_uids = good_uids[:args.num_samples]
    uids, images, captions = extract_samples_from_tar_files(good_uids, tar_files)
    create_samples_visualization(uids, images, captions, f"Good uids", output_dir )




if __name__ == "__main__":
    main(cli())