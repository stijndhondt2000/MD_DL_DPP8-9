#!/bin/bash
set -e
source /project_antwerp/MD_DL_DPP8-9/setup_script.sh

input_dir="/project_antwerp/dataset/safetensors" # absolute path of the safetensors

python3 -m src.cli.preprocess_images --input-dir "$input_dir" --output-dir "$1"