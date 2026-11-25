#!/bin/bash
set -e

input_dir="dataset/safetensors"

python -m src.cli.preprocess_images --input-dir "$input_dir" --output-dir "$1"