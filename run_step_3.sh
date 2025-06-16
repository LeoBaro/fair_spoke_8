#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CUDA_VISIBLE_DEVICES=1,2,3 python "$SCRIPT_DIR/data_quality_pipeline/src/made/bin/main.py" \
    --filtering-step-name MultimodalAlignmentFilter \
    --shards-path "/davinci-1/work/lbaroncelli/fair_spoke_8/full_datacomp_unimodal_vision_filter_output" \
    --config-path "$SCRIPT_DIR/config_step_3.yaml" \
    --log-folder ./full_datacomp_multimodal_alignment_filter_logs \
    --output-folder ./full_datacomp_multimodal_alignment_filter_output