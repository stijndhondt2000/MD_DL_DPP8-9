# !/bin/bash

# call test(model) script for SCNN
checkpoint_path="/project_antwerp/MD_DL_DPP8-9/output/models/SCNN/20261020-134127/SCNN.pth"
splits="validation,test"
python3 -m src.cli.test_model --checkpoint $checkpoint_path --splits $splits
