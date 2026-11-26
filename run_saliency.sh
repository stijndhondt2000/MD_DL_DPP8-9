#!/bin/bash
set -euo pipefail

source /project_antwerp/MD_DL_DPP8-9/setup_script.sh

checkpoint_path="/project_antwerp/MD_DL_DPP8-9/output/models/DCNN/20261118-203251/current_best_model_epoch_6.pth"
output_dir="${1:?Usage: bash run_saliency.sh /path/to/new_output_dir}"
image_dir="project_antwerp/MD_DL_DPP8-9/dataset/images"


python3 -m src.cli.saliency_images \
  --image-dir "$image_dir" \
  --checkpoint "$checkpoint_path" \
  --output-dir "$output_dir" \
  --max-samples 2 \
  2>&1
