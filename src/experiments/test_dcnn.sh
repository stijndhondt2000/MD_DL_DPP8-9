# !/bin/bash

# call test(model) script for CustomDenseNet
checkpoint_path="/project_antwerp/MD_DL_DPP8-9/output/models/DCNN/20261118-203251/current_best_model_epoch_6.pth"
splits="external val train test"
python3 -m src.cli.test_model --checkpoint $checkpoint_path --splits $splits
