import math
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from made.data_pipeline.utils import set_plotting_configuration

def create_output_folder(filtering_step_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(__file__).parent / f"out_{timestamp}_{filtering_step_name.lower()}"
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder

def create_result_file(output_folder):
    result_file = output_folder / "benchmark_results.csv"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("num_workers,batch_size,iteration_index,took\n")
    return result_file

def create_results_and_log_folders(output_folder: Path, suffix: str = ""):
    results_folder = output_folder / f"results_{suffix}"
    log_folder = output_folder / f"logs_{suffix}"
    results_folder.mkdir(exist_ok=True, parents=True)
    log_folder.mkdir(exist_ok=True, parents=True)
    return results_folder, log_folder


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


def create_samples_visualization(
        uids: list[str], 
        images: list[np.ndarray], 
        captions: list[str], 
        title: str, 
        output_path: Path, 
        dpi: int = 100
    ):
    set_plotting_configuration()
    assert len(images) == len(captions), "Input lists must be of equal length"
    
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
            image = images[i].resize((256, 256))
            ax.imshow(image)
            if uids[i] is None:
                ax_title = f"{captions[i]}"
            else:
                ax_title = f"{uids[i]}\n{captions[i]}"
            ax_title = "\n".join(textwrap.wrap(ax_title, width=35))
            if len(ax_title) > 55:
                ax_title=ax_title[:55]+"[...]"
            ax.set_title(ax_title, fontsize=15)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide any unused subplot axes

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    plt.subplots_adjust(top=0.97, bottom=0.01, left=0.01, right=0.99)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Visualization saved to {output_path}")