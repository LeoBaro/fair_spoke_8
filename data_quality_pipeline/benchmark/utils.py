from datetime import datetime
from pathlib import Path
import numpy as np
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import textwrap


def create_config(config: dict, dump_dir: Path):
    config_raw = f"""
infrastructure:
    enable_metrics: {config["enable_metrics"]}
    save_bad_uids: {config["save_bad_uids"]}
    logging_level: WARNING
    log_to_driver: true
    num_workers: {config["num_workers"]} 
    dump_tar_every_n_samples: 10000
    
unimodal_text:
    batch_size: {config["batch_size"]} 

    caption_min_words: 2
    caption_min_chars: 5

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    tagging_model_name: en_core_web_trf
    good_captions_pos_distribution_path: models/common_pos_patterns.txt

unimodal_vision:
    batch_size: {config["batch_size"]} 

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 3.0
    image_min_dimension: 50

    text_threshold: 0.6
    text_detection_model_path: models
    text_detection_mag_ratio: 0.5

    specificity_threshold: 0.5           
    curvature: 1.0    

multimodal:
    batch_size: {config["batch_size"]} 

    dfn_model: leobaro/DFN-public
    dfn_percentile_to_drop: 25
    clip_caption_max_length: 77

specificity:
    batch_size: {config["batch_size"]} 

    model_path: models/ckpt.pt
    reference_path: models/reference.pt
    specificity_threshold: 0.27
"""
    
    config_file_path = dump_dir / "single_node_config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write(config_raw)
    return config_file_path

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