# !/bin/bash

# call test(model) script for CustomDenseNet
checkpoint_path=v
splits="external test"
python3 -m src.cli.test_model --checkpoint $checkpoint_path --splits $splits
