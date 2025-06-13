#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python "$SCRIPT_DIR/data_quality_pipeline/src/made/bin/main.py" \
    --filtering-step-name UnimodalTextFilter \
    --shards-path "$SCRIPT_DIR/data_quality_pipeline/benchmark/data" \
    --config-path "$SCRIPT_DIR/config.yaml" \
    --log-folder ./logs \
    --output-folder ./output