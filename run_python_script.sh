#!/bin/bash
source /project_antwerp/MD_DL_DPP8-9/setup_script.sh
# take the first argument as the filename of the python script to run
PYTHON_SCRIPT=$1
python3 "$PYTHON_SCRIPT"
