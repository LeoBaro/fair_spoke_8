import os
import yaml
from pathlib import Path

from made.data_pipeline.common.singleton import Singleton
from made.data_pipeline.common.dict_wrapper import DictWrapper

class Config(metaclass=Singleton):

    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)
        for key, value in self._config.items():
            setattr(self, key, DictWrapper(value) if isinstance(value, dict) else value)

    def __str__(self):
        return yaml.dump(self._config, default_flow_style=False)
    
    @staticmethod
    def create_config(override_config: dict, output_path: Path):
        config_raw = f"""
infrastructure:
    enable_metrics: {override_config["enable_metrics"] if "enable_metrics" in override_config else True}
    save_bad_uids:  {override_config["save_bad_uids"] if "save_bad_uids" in override_config else True}
    logging_level:  {override_config["logging_level"] if "logging_level" in override_config else "INFO"}
    log_to_driver:  {override_config["log_to_driver"] if "log_to_driver" in override_config else True}
    num_workers:    {override_config["num_workers"] if "num_workers" in override_config else 1} 
    num_gpus_per_worker: {override_config["num_gpus_per_worker"] if "num_gpus_per_worker" in override_config else 0.1}
    batch_size:          {override_config["batch_size"] if "batch_size" in override_config else 50} 
    dump_tar_every_n_samples: {override_config["dump_tar_every_n_samples"] if "dump_tar_every_n_samples" in override_config else 10000}
    ray_object_store_memory: {override_config["ray_object_store_memory"] if "ray_object_store_memory" in override_config else 2e10}
    
unimodal_text:
    caption_min_words: {override_config["caption_min_words"] if "caption_min_words" in override_config else 2}
    caption_min_chars: {override_config["caption_min_chars"] if "caption_min_chars" in override_config else 5}

    lang_detection_model_path:      {override_config["lang_detection_model_path"] if "lang_detection_model_path" in override_config else "models/lid.176.bin"}
    lang_detection_score_threshold: {override_config["lang_detection_score_threshold"] if "lang_detection_score_threshold" in override_config else 0.7}
    lang_detection_language:        {override_config["lang_detection_language"] if "lang_detection_language" in override_config else "en"}

    tagging_model_name:                  {override_config["tagging_model_name"] if "tagging_model_name" in override_config else "en_core_web_trf"}
    good_captions_pos_distribution_path: {override_config["good_captions_pos_distribution_path"] if "good_captions_pos_distribution_path" in override_config else "models/common_pos_patterns.txt"}

unimodal_vision:
    image_min_aspect_ratio: {override_config["image_min_aspect_ratio"] if "image_min_aspect_ratio" in override_config else 0.8}
    image_max_aspect_ratio: {override_config["image_max_aspect_ratio"] if "image_max_aspect_ratio" in override_config else 3.0}
    image_min_dimension:    {override_config["image_min_dimension"] if "image_min_dimension" in override_config else 50}

    text_threshold:            {override_config["text_threshold"] if "text_threshold" in override_config else 0.6}
    text_detection_model_path: {override_config["text_detection_model_path"] if "text_detection_model_path" in override_config else "models"}
    text_detection_mag_ratio:  {override_config["text_detection_mag_ratio"] if "text_detection_mag_ratio" in override_config else 0.5}

multimodal:
    dfn_model:                         {override_config["dfn_model"] if "dfn_model" in override_config else "leobaro/DFN-public"}
    dfn_similarity_score_threshold:    {override_config["dfn_similarity_score_threshold"] if "dfn_similarity_score_threshold" in override_config else 4.8}
    clip_caption_max_length:           {override_config["clip_caption_max_length"] if "clip_caption_max_length" in override_config else 77}

specificity:
    model_path: {override_config["model_path"] if "model_path" in override_config else "models/ckpt.pt"}
    reference_path: {override_config["reference_path"] if "reference_path" in override_config else "models/reference.pt"}
    specificity_threshold: {override_config["specificity_threshold"] if "specificity_threshold" in override_config else 0.27}
    weight: {override_config["weight"] if "weight" in override_config else 0.5}

semdedup:
    model_name: {override_config["semdedup_model_name"] if "semdedup_model_name" in override_config else "openai/clip-vit-large-patch14"}
    # -- directories
    save_folder: {override_config["semdedup_save_folder"] if "semdedup_save_folder" in override_config else "data_quality_pipeline/src/made/models/semdedup/data/"}
    sorted_clusters_path: {override_config["semdedup_sorted_clusters_path"] if "semdedup_sorted_clusters_path" in override_config else "data_quality_pipeline/src/made/models/semdedup/data/sorted_clusters"}
    semdedup_pruning_tables_path: {override_config["semdedup_pruning_tables_path"] if "semdedup_pruning_tables_path" in override_config else "data_quality_pipeline/src/made/models/semdedup/data/dataframes"}
    embs_memory_loc: {override_config["semdedup_embs_memory_loc"] if "semdedup_embs_memory_loc" in override_config else "data_quality_pipeline/src/made/models/semdedup/data/embeddings/embs.npy"}
    path_memory_loc: {override_config["semdedup_path_memory_loc"] if "semdedup_path_memory_loc" in override_config else "data_quality_pipeline/src/made/models/semdedup/data/embeddings/path.npy"}
    # -- data type
    paths_str_type: {override_config["semdedup_paths_str_type"] if "semdedup_paths_str_type" in override_config else "'U256'"}
    embed_float_type: {override_config["semdedup_embed_float_type"] if "semdedup_embed_float_type" in override_config else "'float32'"}
    # -- data loader
    num_workers: {override_config["semdedup_num_workers"] if "semdedup_num_workers" in override_config else 0}
    # -- dataset size
    dataset_size: {override_config["semdedup_dataset_size"] if "semdedup_dataset_size" in override_config else 0}
    batch_size: {override_config["semdedup_batch_size"] if "semdedup_batch_size" in override_config else 16}
    # -- embeddings size
    emd_size: {override_config["semdedup_emd_size"] if "semdedup_emd_size" in override_config else 768}
    # -- Clustering parameters
    clustering:
        num_clusters: {override_config["semdedup_clustering_num_clusters"] if "semdedup_clustering_num_clusters" in override_config else 500}
        niter: {override_config["semdedup_clustering_niter"] if "semdedup_clustering_niter" in override_config else 1000}
        keep_hard: {override_config["semdedup_clustering_keep_hard"] if "semdedup_clustering_keep_hard" in override_config else True}
        sim_metric: {override_config["semdedup_clustering_sim_metric"] if "semdedup_clustering_sim_metric" in override_config else "'cosine'"}
        Kmeans_with_cos_dist: {override_config["semdedup_clustering_Kmeans_with_cos_dist"] if "semdedup_clustering_Kmeans_with_cos_dist" in override_config else True}
        save_folder: {override_config["semdedup_clustering_save_folder"] if "semdedup_clustering_save_folder" in override_config else "data_quality_pipeline/src/made/models/semdedup/data/clustering"}
        text_emb_memory_loc: {override_config["semdedup_clustering_text_emb_memory_loc"] if "semdedup_clustering_text_emb_memory_loc" in override_config else None}
    # -- seed
    seed: {override_config["semdedup_seed"] if "semdedup_seed" in override_config else 1234}
    # -- largest cluster size the memory is large enough to process. If the cluster size is larger than it, we will devide the cluster into small clusters and process each one separately.
    largest_cluster_size_to_process: {override_config["semdedup_largest_cluster_size_to_process"] if "semdedup_largest_cluster_size_to_process" in override_config else 10000000}
    eps: {override_config["semdedup_eps"] if "semdedup_eps" in override_config else 0.1}
    eps_list: {override_config["semdedup_eps_list"] if "semdedup_eps_list" in override_config else [
        0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 
        0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 
        0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 
        0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
        0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 
        0.27, 0.28, 0.29, 0.3, 0.32, 0.34, 0.36, 
        0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 
        1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4,
        2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 
        4.0, 4.2, 4.4, 4.6, 4.8, 5.0
        ]}
    which_to_keep: {override_config["semdedup_which_to_keep"] if "semdedup_which_to_keep" in override_config else "'easy'"}
    # -- output
    output_txt_path: {override_config["semdedup_output_txt_path"] if "semdedup_output_txt_path" in override_config else "data_quality_pipeline/src/made/models/semdedup/data/kept_examples.txt"}
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(config_raw)
        return output_path