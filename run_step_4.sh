#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1,3 python "$SCRIPT_DIR/data_quality_pipeline/src/made/bin/main.py" \
    --filtering-step-name MultimodalSpecificityFilter \
    --shards-path "/davinci-1/work/lbaroncelli/fair_spoke_8/full_datacomp_multimodal_alignment_filter_output" \
    --config-path "$SCRIPT_DIR/config_step_4.yaml" \
    --log-folder ./full_datacomp_multimodal_specificity_filter_logs \
    --output-folder ./full_datacomp_multimodal_specificity_filter_output