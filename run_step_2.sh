#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python "$SCRIPT_DIR/data_quality_pipeline/src/made/bin/main.py" \
    --filtering-step-name UnimodalVisionFilter \
    --shards-path "/davinci-1/work/lbaroncelli/fair_spoke_8/full_datacomp_unimodal_text_filter_output" \
    --config-path "$SCRIPT_DIR/config_step_2.yaml" \
    --log-folder ./full_datacomp_unimodal_vision_filter_logs \
    --output-folder ./full_datacomp_unimodal_vision_filter_output