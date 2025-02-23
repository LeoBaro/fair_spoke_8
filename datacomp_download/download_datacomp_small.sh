#!/bin/bash

processes_count=8 # each process download one shard (about 510K images)
thread_count=128 # ..with 4 threads

# Upgrade huggingface-hub to v0.13.4 ! 
# pip install huggingface-hub==0.13.4

# Each time this script is run, it will overwrite the shards directory,

python src/datacomp_download/download_upstream.py \
--scale small \
--data_dir /home/leobaro/Downloads/datasets/web/datacomp \
--processes_count $processes_count \
--thread_count $thread_count \
--retries 5 \
--enable_wandb \
--skip_metadata
#--skip_shards \
#--overwrite_metadata


