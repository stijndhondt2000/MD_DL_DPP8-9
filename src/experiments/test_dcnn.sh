# !/bin/bash

# call test(model) script for CustomDenseNet
checkpoint_path="project_antwerp/MD_DL_DPP8-9/output/models/CustomDenseNet_Randomsplit_Dataset/20260805-025118/CustomDenseNet_Randomsplit_Dataset.pth"
splits="external val train test"
python3 -m src.cli.test_model --checkpoint $checkpoint_path --splits $splits
