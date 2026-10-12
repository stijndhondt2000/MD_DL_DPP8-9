#!/bin/bash
# cd to the root directory of the project
cd /project_antwerp/MD_DL_DPP8-9 || exit
token=$(cat '/project_antwerp/token.txt')
echo "machine github.com login oauth2 password ${token}" >> ~/.netrc
git pull
# take the first argument as the filename of the python script to run
PYTHON_SCRIPT=$1
python3 "$PYTHON_SCRIPT"
