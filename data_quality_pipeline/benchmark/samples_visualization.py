import argparse
import glob
import os
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import math
from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

def get_good_uids(results_folder):
    good_uids = glob.glob(os.path.join(results_folder, "*.npy"))
    good_uids = np.load(good_uids[0])
    good_uids = [f"{a:016x}{b:016x}" for a, b in good_uids]
    return good_uids

def get_bad_uids(bad_uids_file: str):
    bad_uids = json.load(open(bad_uids_file, encoding="utf-8"))
    return bad_uids

def extract_samples_from_tar_files(uids: list[str], tar_files: list[str], batch_size: int = 500, num_samples: int = 20):
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=batch_size,
        valid_uids=uids
    )   
    uids, images, captions = get_next_batch(iter(dataset)) 
    sample_idx = np.random.choice(len(images), num_samples, replace=False)
    images = [images[i] for i in sample_idx]
    captions = [captions[i] for i in sample_idx]
    uids = [uids[i] for i in sample_idx]
    return uids, images, captions

def create_samples_visualization(uids: list[str], images: list[np.ndarray], captions: list[str], title: str, output_dir: str):
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{title.replace(' ', '_')}.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Visualization saved to {filepath}")



def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results-folder", type=str, required=True)
    parser.add_argument("-t", "--tar-files-path", type=str, required=False, default="/home/leobaro/workspace/labs/fair_spoke_8/data_quality_pipeline/benchmark/data")
    return parser.parse_args()


def main(args):
    results_folder = glob.glob(os.path.join(args.results_folder, "results__*"))[0] # take the first one
    tar_files = collect_tar_files(args.tar_files_path, recursive=False)

    bad_uids = glob.glob(os.path.join(results_folder, "bad_uids_*.json"))
    for bad_uid in bad_uids:
        bad_uid_json = get_bad_uids(bad_uid)
        for filter_name, bad_uids_samples in bad_uid_json.items():
            uids, images, captions = extract_samples_from_tar_files(bad_uids_samples, tar_files)
            create_samples_visualization(uids, images, captions, f"Bad uids for {filter_name} filter", Path(results_folder) / "bad_uids_visualizations" )

    good_uids = get_good_uids(results_folder)
    uids, images, captions = extract_samples_from_tar_files(good_uids, tar_files)
    create_samples_visualization(uids, images, captions, f"Good uids", Path(results_folder) / "good_uids_visualizations" )




if __name__ == "__main__":
    main(cli())