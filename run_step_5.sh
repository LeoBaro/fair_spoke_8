#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/data_quality_pipeline/src/made/bin/main.py" \
    --filtering-step-name SemanticDedupFilter \
    --shards-path "/home/leobaro/Downloads/datasets/web/datacomp/full_datacomp_multimodal_specificity_filter_output_no_duplicates" \
    --config-path "$SCRIPT_DIR/config_step_5.yaml" \
    --log-folder ./full_datacomp_semantic_dedup_filter_logs \
    --output-folder ./full_datacomp_semantic_dedup_filter_output