import logging
from datetime import datetime
from pathlib import Path

import pytest
import ray

from made.config import Config


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def output_folder():
    return Path(__file__).parent / "out"

@pytest.fixture(scope="session")
def log_folder():
    log_folder = Path(__file__).parent / "logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder

@pytest.fixture(scope="session")
def tar_files(data_path):
    return sorted([str(data_path/s) for s in Path(data_path).glob("*.tar")])

def pytest_addoption(parser):
    parser.addoption(
        "--ray", action="store_true", default=False, help="Run Ray-based tests"
    )
    # For future multi node tests 
    parser.addoption(
        "--multinode", action="store_true", default=False, help="Run multinode Ray-based tests"
    )

@pytest.fixture(scope="session")
def ray_flag(request):
    return request.config.getoption("--ray")

# For future multi node tests 
@pytest.fixture(scope="session")
def multinode_flag(request):
    return request.config.getoption("--multinode")

@pytest.fixture(scope="session")
def ray_init(ray_flag):
    if ray_flag:
        ray.init(
            num_cpus=4,
            num_gpus=1,
            logging_level=logging.DEBUG,
            runtime_env={
                "env_vars": {
                    "RAY_DEBUG": "1"
                }
        })
    yield
    if ray_flag:
        ray.shutdown()

@ray.remote
def post_mortem(x):
    x += 1
    raise Exception("An exception is raised")
    return x

@pytest.fixture(scope="session")
def config_path(request):
    config_file_for_tests_path = "/tmp/test.yaml"
    with open(config_file_for_tests_path, "w") as f:
        f.write("""
infrastructure:
    num_workers: 2
    enable_metrics: true
    save_filtered_uids: false
    logging_level: DEBUG
    apply_filters: true

unimodal:
    batch_size: 50

    caption_min_words: 2
    caption_min_chars: 5

    lang_detection_model_path: models/lid.176.bin
    lang_detection_score_threshold: 0.7
    lang_detection_language: en

    tagging_model_name: en_core_web_trf
    good_captions_pos_distribution_path: models/common_pos_patterns.txt

    image_min_aspect_ratio: 0.8
    image_max_aspect_ratio: 3.0
    image_min_dimension: 50

    text_threshold: 0.6
    text_detection_model_path: models
    text_detection_mag_ratio: 0.5

    semdedup:
        # -- model
        model_name: "openai/clip-vit-base-patch32"
        # -- directories
        save_folder: "data_quality_pipeline/src/made/semdedup/data/"
        sorted_clusters_path: "data_quality_pipeline/src/made/semdedup/data/sorted_clusters"
        semdedup_pruning_tables_path: "data_quality_pipeline/src/made/semdedup/data/dataframes"
        embs_memory_loc: "data_quality_pipeline/src/made/semdedup/data/embeddings/embs.npy"
        path_memory_loc: "data_quality_pipeline/src/made/semdedup/data/embeddings/path.npy"
        # -- data type
        paths_str_type: 'U256'
        embed_float_type: 'float32'
        # -- data loader
        num_workers: 0
        # -- dataset size
        dataset_size: 0
        batch_size: 16
        # -- embeddings size
        emd_size: 512
        # -- Clustering parameters
        clustering:
            num_clusters: 50 #50000
            niter: 100
            keep_hard: True # True for hard examples
            sim_metric: 'cosine' # choose form ['cosine', 'l2']
            Kmeans_with_cos_dist: True # True for using cosine similarity for kmeans clustering
            save_folder: "data_quality_pipeline/src/made/semdedup/data/clustering"
            text_emb_memory_loc: None
        # -- seed
        seed: 1234
        # -- largest cluster size the memory is large enough to process. If the cluster size is larger than it, we will devide the cluster into small clusters and process each one separately.
        largest_cluster_size_to_process: 10000000
        eps: 3.0
        eps_list: [
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
            ]
        which_to_keep: "easy"
        # -- output
        output_txt_path: "data_quality_pipeline/src/made/semdedup/data/kept_examples.txt"
multimodal:
    batch_size: 256
    dfn_model: leobaro/DFN-public
    dfn_percentile_to_drop: 25
    clip_caption_max_length: 77

""")
    return config_file_for_tests_path


@pytest.fixture(scope="session")
def config(config_path):
    return Config(config_path)

