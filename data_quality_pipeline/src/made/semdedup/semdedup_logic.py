import math
import os
import pickle
import pprint
import random
import time
import torch

import numpy as np
import pandas as pd

from constants import DIST_METRIC_INDEX, IMAGE_ID_IN_CLUSTER_INDEX, IMAGE_NAME_INDEX
from my_utils import load_config
from tqdm import tqdm

config = load_config("semdedup_configs.yaml")

def init_memmap_embs(
    embs_memory_loc: str, 
    dataset_size: int, 
    emd_size: int = 512, 
    dtype: str = "float32"
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        embs_memory_loc, 
        dtype=dtype, mode="r", 
        shape=(dataset_size, emd_size)
    )
    return embs

def contains_duplicates(arr):
    return len(np.unique(arr)) != len(arr)

def semdedup(cluster, cluster_reps):
    start_time = time.time()
    ## -- compute pairwise cos sim between cluster items, 
    ## then replace to diagonal with zeros to ignore self similarity
    pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
    pair_w_sim_matrix.fill_diagonal_(0.0)
    assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

    ## -- get paths to cluster i images
    image_urls = cluster[:, IMAGE_NAME_INDEX]

    ## -- make sure all the paths are unique this ensure that the duplicates 
    # are really stored many time times on memory
    assert not contains_duplicates(image_urls)

    ## -- 2) compute the sum of all pairwise sim values exept the diagonal 
    # (diagonal items = 1.0)
    avg_sim_to_others_list = (1 / (pair_w_sim_matrix.shape[0] - 1)) * (
        torch.sum(pair_w_sim_matrix, dim=0)
    )  # -- array of shape (cluster_size x 1)

    ## -- 3) compute max pairwise similarity
    max_pair_w_sim_list = torch.max(pair_w_sim_matrix, dim=0)[
        0
    ]  # -- array of shape (cluster_size x 1)
    min_pair_w_sim_list = torch.min(pair_w_sim_matrix, dim=0)[
        0
    ]  # -- array of shape (cluster_size x 1)
    std_pair_w_sim = pair_w_sim_matrix.std()

    ## -- 4) average value of cos similarity to cluster centroid
    avg_sim_to_cent = (1 - cluster[:, DIST_METRIC_INDEX].astype("float32")).mean()
    std_sim_to_cent = (1 - cluster[:, DIST_METRIC_INDEX].astype("float32")).std()

    ## -- We need upper tringular matrix because 
    # (1) we don't need to look at self sim (always=1)
    # (2) we need the compinations not permutations
    triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
    del pair_w_sim_matrix
    # pair_w_sim_matrix[lower_tri_ids] = 0

    ## -- if the max sim between one example and any other example is > 1-eps, 
    # remove this example
    M = torch.max(triu_sim_mat, dim=0)[0]
    print(f"Step time: {time.time()-start_time}(s)")

    return (
        M,
        avg_sim_to_others_list,
        max_pair_w_sim_list,
        min_pair_w_sim_list,
        std_pair_w_sim,
        avg_sim_to_cent,
        std_sim_to_cent,
    )

def process_shard(shard: int, config=None):
    # Use provided config or import from module if not provided
    if config is None:
        raise ValueError("Config must be provided to process_shard function")
        
    # print("SemDeDup params: ", config)
    start_time = time.time()
    end_shard = config["num_clusters"]
    print(f"This process will process clusters {shard} to {end_shard}")

    # For a single-node run, process the entire shard without task-level division.
    start = shard
    end = end_shard
    print(f"Processing clusters from {start} to {end}")

    embs = init_memmap_embs(
        config["embs_memory_loc"], 
        config["dataset_size"],
        config["emd_size"]
    )
    statistics_df = pd.DataFrame(
        columns=[
            "cluster_size",
            "cluster_id",
            "avg_sim_to_cent",
            "std_sim_to_cent",
            "std_pair_w_sim",
            "avg_sim_to_others_list",
            "max_pair_w_sim_list",
            "min_pair_w_sim_list",
        ]
    )

    eps_df_dicts = {
        eps: pd.DataFrame(
            columns=["duplicates_ratio", "num_duplicates", "cluster_id"]
        )
        for eps in config["eps_list"]
    }

    eps_dict_file_loc = os.path.join(
        config["save_folder"], f"statistics/dicts/shard_{start}.pt"
    )
    statistics_df_file_loc = os.path.join(
        config["save_folder"], f"statistics/dataframes/shard_{start}.pkl"
    )

    step_time = []

    for cluster_id in tqdm(range(start, end)):
        step_start_time = time.time()

        df_file_loc = os.path.join(
            config["save_folder"], f"dataframes/cluster_{cluster_id}.pkl"
        )

        if os.path.exists(df_file_loc):
            print(f"{df_file_loc} exists, moving on")
            continue

        # Load cluster representations.
        cluster_i = np.load(
            os.path.join(
                config["sorted_clusters_path"], f"cluster_{cluster_id}.npy"
            )
        )
        cluster_size = cluster_i.shape[0]
        print("cluster_size: ", cluster_size)

        if cluster_size == 1:
            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = [0]
            points_to_remove_df["image_id_in_dataset"] = cluster_i[
                :, IMAGE_ID_IN_CLUSTER_INDEX
                ]
            points_to_remove_df["cluster_url"] = cluster_i[:, IMAGE_NAME_INDEX]
            for eps in config["eps_list"]:
                points_to_remove_df[f"eps={eps}"] = [False]
            if config["save_folder"] != "":
                df_dir = os.path.dirname(df_file_loc)
                os.makedirs(df_dir, exist_ok=True)
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)
            print("DONE cluster_id ", cluster_id)
            continue

        # Decide which cluster examples to keep.
        clutser_items_indices = list(range(cluster_size))
        if config["which_to_keep"].lower() == "random":
            random.shuffle(clutser_items_indices)
            cluster_i = cluster_i[clutser_items_indices]
        elif config["which_to_keep"].lower() == "easy":
            clutser_items_indices = clutser_items_indices[::-1]
            cluster_i = cluster_i[clutser_items_indices]


        # Get indices for cluster items in the dataset.
        cluster_ids = cluster_i[:, IMAGE_ID_IN_CLUSTER_INDEX].astype("int32")
        cluster_urls = cluster_i[:, IMAGE_NAME_INDEX].astype("str")
        cluster_reps = embs[cluster_ids]
        cluster_reps = torch.tensor(cluster_reps)

        # Initialize tensors and variables for statistics.
        avg_sim_to_others_list = torch.tensor([])
        max_pair_w_sim_list = torch.tensor([])
        min_pair_w_sim_list = torch.tensor([])
        std_pair_w_sim = 0
        avg_sim_to_cent = 0
        std_sim_to_cent = 0
        M = torch.tensor([])

        # Process cluster in smaller chunks if needed.
        num_small_clusters = (
            math.ceil(cluster_size / config["largest_cluster_size_to_process"]) + 1
        )
        cluster_part_ids = np.linspace(
            0, cluster_size, num_small_clusters, dtype="int64"
        )
        for i in range(len(cluster_part_ids) - 1):
            (
                tem_M,
                tem_avg_sim_to_others_list,
                tem_max_pair_w_sim_list,
                tem_min_pair_w_sim_list,
                tem_std_pair_w_sim,
                tem_avg_sim_to_cent,
                tem_std_sim_to_cent,
            ) = semdedup(
                cluster_i,
                cluster_reps[cluster_part_ids[i] : cluster_part_ids[i + 1]],
            )

            avg_sim_to_others_list = torch.cat(
                (avg_sim_to_others_list, tem_avg_sim_to_others_list)
            )
            max_pair_w_sim_list = torch.cat(
                (max_pair_w_sim_list, tem_max_pair_w_sim_list)
            )
            min_pair_w_sim_list = torch.cat(
                (min_pair_w_sim_list, tem_min_pair_w_sim_list)
            )
            std_pair_w_sim += tem_std_pair_w_sim
            avg_sim_to_cent = tem_avg_sim_to_cent
            std_sim_to_cent = tem_std_sim_to_cent
            M = torch.cat((M, tem_M))

        std_pair_w_sim /= len(cluster_part_ids)
        points_to_remove_df = pd.DataFrame()
        points_to_remove_df["indices"] = clutser_items_indices
        points_to_remove_df["image_id_in_dataset"] = cluster_ids
        points_to_remove_df["cluster_url"] = cluster_urls

        for eps in config["eps_list"]:
            eps_points_to_remove = M > 1 - eps
            points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

            eps_num_duplicates = sum(eps_points_to_remove).item()
            eps_duplicates_ratio = 100 * eps_num_duplicates / cluster_size

            eps_df_dicts[eps] = pd.concat(
                [
                    eps_df_dicts[eps],
                    pd.DataFrame(
                        {
                            "duplicates_ratio": eps_duplicates_ratio,
                            "num_duplicates": eps_num_duplicates,
                            "cluster_id": cluster_id,
                        },
                        index=range(cluster_size),
                    ),
                ]
            )

        statistics_df = pd.concat(
            [
                statistics_df,
                pd.DataFrame(
                    {
                        "cluster_size": cluster_size,
                        "cluster_id": cluster_id,
                        "avg_sim_to_cent": avg_sim_to_cent,
                        "std_sim_to_cent": std_sim_to_cent,
                        "std_pair_w_sim": std_pair_w_sim,
                        "avg_sim_to_others_list": [avg_sim_to_others_list],
                        "max_pair_w_sim_list": [max_pair_w_sim_list],
                        "min_pair_w_sim_list": [min_pair_w_sim_list],
                    }
                ),
            ]
        )

        if config["save_folder"] != "":
            with open(df_file_loc, "wb") as file:
                pickle.dump(points_to_remove_df, file)

        step_time.append(time.time() - step_start_time)
        print("Step time so far:", step_time)
        print("DONE cluster:", cluster_id)

    if config["save_folder"] != "":
        eps_dir = os.path.dirname(eps_dict_file_loc)
        os.makedirs(eps_dir, exist_ok=True)
        torch.save(eps_df_dicts, eps_dict_file_loc)

        stats_dir = os.path.dirname(statistics_df_file_loc)
        os.makedirs(stats_dir, exist_ok=True)
        with open(statistics_df_file_loc, "wb") as file:
            pickle.dump(statistics_df, file)

    print("All clusters processed. Step times:", step_time)
    avg_step_time = (sum(step_time) / len(step_time)) if len(step_time) > 0 else 0
    print(
        f"DONE in {((time.time()-start_time)/60):.2f} minutes, Average Step time {avg_step_time:.2f} sec"
    )
    return