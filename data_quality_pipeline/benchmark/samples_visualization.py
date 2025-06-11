import argparse
import glob
import os
import numpy as np
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import math
from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from collections import defaultdict

def get_good_uids(results_folder):
    good_uids = glob.glob(os.path.join(results_folder, "*.npy"))
    good_uids = np.load(good_uids[0])
    good_uids = [f"{a:016x}{b:016x}" for a, b in good_uids]
    return good_uids

def extract_samples_from_tar_files(
        uids: list[str], 
        tar_files: list[str], 
        num_samples: int = 20,
        batch_size: int = 5000, 
        get_images: bool = True,
        get_captions: bool = True,
    ):
    assert num_samples <= batch_size, "num_samples must be less than or equal to batch_size"
    dataset = decode_webdataset(
        tar_files,
        get_images=get_images,
        get_captions=get_captions,
        batch_size=batch_size,
        valid_uids=uids
    )
    uids, images, captions = [], [], []
    if get_images and get_captions:
        uids, images, captions = get_next_batch(iter(dataset)) 
    elif get_images:
        uids, images = get_next_batch(iter(dataset)) 
    elif get_captions:
        uids, captions = get_next_batch(iter(dataset)) 
    sample_idx = np.random.choice(len(uids), num_samples, replace=False)
    if get_images:
        images = [images[i] for i in sample_idx]
    if get_captions:
        captions = [captions[i] for i in sample_idx]
    uids = [uids[i] for i in sample_idx]
    return uids, images, captions

def create_samples_visualization(uids: list[str], images: list[np.ndarray], captions: list[str], title: str, output_dir: Path):
    sns.set_theme(style="darkgrid")

    assert len(uids) == len(images) == len(captions), "Input lists must be of equal length"
    
    num_samples = len(images)
    num_cols = 4
    num_rows = math.ceil(num_samples / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    fig.suptitle(title, fontsize=16)

    # Flatten axes array for easy iteration
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_rows * num_cols):
        ax = axes[i]
        if i < num_samples:
            ax.imshow(images[i])
            wrapped_caption = "\n".join(textwrap.wrap(captions[i], width=40))
            wrapped_uid = "\n".join(textwrap.wrap(f"UID: {uids[i]}", width=40))
            ax.set_title(f"{wrapped_uid}\n{wrapped_caption}", fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide any unused subplot axes

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    filepath = output_dir / f"{title.replace(' ', '_')}.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Visualization saved to {filepath}")

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
        uids, images, captions = extract_samples_from_tar_files(bad_uids_samples, tar_files, get_images=False, get_captions=True, num_samples=500)
        create_txt_file_list(uids, captions, Path(args.folder) / "bad_uids_visualizations" / f"{filter_name}.txt")
        
        uids, images, captions = extract_samples_from_tar_files(bad_uids_samples, tar_files, get_images=True, get_captions=True, num_samples=20)
        create_samples_visualization(uids, images, captions, f"Bad uids for {filter_name} filter", output_dir )


    # good_uids = get_good_uids(results_folder)
    # uids, images, captions = extract_samples_from_tar_files(good_uids, tar_files)
    # create_samples_visualization(uids, images, captions, f"Good uids", Path(results_folder) / "good_uids_visualizations" )




if __name__ == "__main__":
    main(cli())