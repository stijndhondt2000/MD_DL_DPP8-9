"""
This script orchestrates the complete data preprocessing pipeline for the molecular dataset.
It performs the following steps:
1. Generates a dataset from TAR files, converting them into disk-backed dataset shards.
2. Loads the dataset shards, concatenates them into a single dataset, shuffles the rows, and applies the project-specific train/val/test split.
3. Prepares the final dataset for model training by converting it into safetensors format, ensuring efficient storage and retrieval.
The pipeline is designed to be modular, allowing for easy adjustments to the preprocessing steps as needed.
"""

import os

from src.infrastructure.dotenvManager import dotenvManager
from src.data.data_preprocessing.pre_processing import generate_development_dataset_from_tars
from src.data.data_preprocessing.prepare_safetensors import prepare_safetensors
from src.utils.utils import calculate_num_cpus

if __name__ == "__main__":
    dotenv_manager = dotenvManager()
    tar_folder = dotenv_manager.get_compressed_dataset_directory()
    dataset_path = dotenv_manager.get_dataset_directory()
    safetensors_path = dotenv_manager.get_safetensors_directory()
    num_cpus = calculate_num_cpus()
    num_proc = min(1, num_cpus//2) # hyperthreading 

    # Step 1: Generate the full dataset from TAR files
    # 2 cpus do 1 targz file in ~30 minutes. ~130 targz files in total = ~65 hours. 
    # To speed up the process, we can use 32 cpus to process multiple targz files in parallel. = ~4 hours.
    # generate_development_dataset_from_tars(regenerate=True, num_proc=num_proc, skip_existing=True, remove_temp_dataset=False)

    # Step 2: Generate the external dataset 

 
    # Step 3: Prepare the dataset for model training by converting it into safetensors format
    prepare_safetensors(dataset_path=dataset_path, safetensors_path=safetensors_path, splits=['train', 'val', 'test'], skip_existing_cache=True, redo_cache=False, num_proc=num_proc)
