"""Build PyTorch dataloaders from the project’s cached safetensor splits."""

import multiprocessing as mp

from src.data.data_loading.SafetensorsDataset import ShardedSafetensorsDataset
mp.set_start_method('spawn', force=True)
import shutil

import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from src.infrastructure.dotenvManager import dotenvManager
from src.config.training_config import TrainingConfig
from src.infrastructure.logger import get_logger
logging = get_logger()


def _effective_worker_count(requested_cpus: int) -> int:
    """Use physical-core-like worker count on hyperthreaded systems."""
    requested = max(1, int(requested_cpus))
    return max(1, requested // 2)


# Define a type for the dataloader dict
DataLoaderDict = dict[str, torch.utils.data.DataLoader]

def initialize_dataloaders(config: TrainingConfig, dotenv_manager: dotenvManager= None, splits=None) -> DataLoaderDict:
    """Create one PyTorch dataloader per requested dataset split.

    The cache must contain sharded safetensors with ``data``, ``labels``, and
    ``num_atoms`` fields. ``config.dataset_size`` limits each split, while
    ``config.batch_size`` and ``config.num_cpus`` control batching and worker
    count. ``keep_all_columns`` are retained for API
    compatibility with older loader implementations.

    Returns:
        Mapping from split names such as ``train`` and ``validation`` to
        configured ``torch.utils.data.DataLoader`` instances.
    """
    if splits is None:
        splits = ['train', 'validation', 'test']
    if dotenv_manager is None:
        dotenv_manager = dotenvManager()

    logging.info("\t...initializing_dataloader complete")
    logging.info("\tPre-encoding labels and packing coordinates into fixed-size tensors in dataset artifacts...")
    requested_workers = max(1, int(config.num_cpus))
    safetensors_directory = dotenv_manager.get_safetensors_directory()
    

    logging.info(f"\tLoading cached safetensors from {safetensors_directory}...")
    requested_workers = max(1, int(config.num_cpus))
    dataloader_workers = _effective_worker_count(requested_workers)
    logging.info(
        f"\tUsing {dataloader_workers} DataLoader workers for packed large-tensor batches "
        f"(requested {requested_workers}, hyperthread-aware)."
    )
    dataloader_args = {
        "batch_size": config.batch_size,
        "num_workers": dataloader_workers,
        "persistent_workers": dataloader_workers > 0,
        "pin_memory": True,  # ← Enable for faster CPU→GPU transfer
        "prefetch_factor": 4 if dataloader_workers > 0 else None,
    }
    dataloader_dict = {}
    for split in splits:
        logging.debug(f"\t\tLoading cached safetensors for {split} split from {safetensors_directory}...")
        logging.debug(f"Creating SafetensorsDataset for {split} split with batch size {config.batch_size} and num_workers {dataloader_workers}, fraction {config.dataset_size}...")
        ds = ShardedSafetensorsDataset(safetensors_directory, split, fraction=config.dataset_size)
        logging.info(f"\t\tCreating dataloader for {split} split with batch size {config.batch_size} and num_workers {dataloader_workers}...")
        dataloader = torch.utils.data.DataLoader(dataset=ds, **dataloader_args)
        dataloader_dict[split] = dataloader

    return dataloader_dict




