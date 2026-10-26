/bin/bash


# call test(model) script for SCNN
checkpoint_path="src/experiments/checkpoints/scnn_checkpoint.pt"
splits="validation,test"
python3 -m src.cli.test_model --checkpoint $checkpoint_path --splits $splits
