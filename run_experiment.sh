#!/bin/bash
source /project_antwerp/MD_DL_DPP8-9/pull_git.sh
# take the first argument as the filename of the python script to run
PYTHON_MODULE=$1
python3 -m "$PYTHON_MODULE"
