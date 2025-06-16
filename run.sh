#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python "$SCRIPT_DIR/data_quality_pipeline/src/made/bin/main.py" \
    --filtering-step-name UnimodalTextFilter \
    --shards-path "/davinci-1/work/lbaroncelli/datacomp" \
    --config-path "$SCRIPT_DIR/config.yaml" \
    --log-folder ./full_datacomp_unimodal_text_filter_logs \
    --output-folder ./full_datacomp_unimodal_text_filter_output