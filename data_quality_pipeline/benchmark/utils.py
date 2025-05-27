from datetime import datetime
from pathlib import Path

def create_config(config: dict, dump_dir: Path):
    config_raw = f"""
infrastructure:
    enable_metrics: {config["enable_metrics"]}
    save_filtered_uids: {config["save_filtered_uids"]}
    logging_level: WARNING
    apply_filters: true
    
unimodal_text:
    num_workers: {config["unimodal_text_num_workers"]} 
    batch_size: {config["text_batch_size"]} 

    caption_min_words: 2
    caption_min_chars: 5

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    tagging_model_name: en_core_web_trf
    good_captions_pos_distribution_path: models/common_pos_patterns.txt

unimodal_vision:
    num_workers: {config["unimodal_vision_num_workers"]} 
    batch_size: {config["vision_batch_size"]} 

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 3.0
    image_min_dimension: 50

    text_threshold: 0.6
    text_detection_model_path: models
    text_detection_mag_ratio: 0.5

    specificity_threshold: 0.5           
    curvature: 1.0    

multimodal:
    num_workers: {config["multimodal_num_workers"]} 
    batch_size: {config["multimodal_batch_size"]}
    dfn_model: leobaro/DFN-public
    dfn_percentile_to_drop: 25
    clip_caption_max_length: 77
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
        f.write("unimodal_text_num_workers,unimodal_vision_num_workers,multimodal_num_workers,text_batch_size,vision_batch_size,multimodal_batch_size,iteration_index,took\n")
    return result_file

def create_results_and_log_folders(output_folder: Path, suffix: str = ""):
    results_folder = output_folder / f"results_{suffix}"
    log_folder = output_folder / f"logs_{suffix}"
    results_folder.mkdir(exist_ok=True, parents=True)
    log_folder.mkdir(exist_ok=True, parents=True)
    return results_folder, log_folder