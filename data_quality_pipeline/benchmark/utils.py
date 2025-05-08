from datetime import datetime
from pathlib import Path

def create_config(num_workers, batch_size):
    config_raw = f"""
infrastructure:
    num_workers: {num_workers} 
    enable_metrics: false
    logging_level: WARNING
    save_npy: false
    apply_filters: false
    
unimodal:
    batch_size: {batch_size} 

    caption_min_words: 2
    caption_min_chars: 5

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    tagging_model_name: en_core_web_trf
    good_captions_pos_distribution_path: models/common_pos_patterns.txt

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 1.8

multimodal:
    batch_size: 32
    clip_model: openai/clip-vit-base-patch32
    clip_score_threshold: 0.3
"""
    
    config_file_path = "/tmp/single_node_config.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write(config_raw)
    return config_file_path

def create_output_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(__file__).parent / f"out_{timestamp}"
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder

def create_results_and_log_folders(output_folder: Path, suffix: str = ""):
    results_folder = output_folder / f"results_{suffix}"
    log_folder = output_folder / f"logs_{suffix}"
    results_folder.mkdir(exist_ok=True, parents=True)
    log_folder.mkdir(exist_ok=True, parents=True)
    return results_folder, log_folder