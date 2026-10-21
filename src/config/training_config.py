from dataclasses import dataclass, field, asdict
import importlib
import os
import sys
import types
from typing import Callable
import json

import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from src.training.batch_preprocessing import prepare_model_batch
from src.transform.tranformators import apply_image_transform, apply_image_transform_noscramble


def calculate_num_cpus():
    """Calculate the number of CPUs to use based on environment variable or default"""
    # in python vs code debug mode, use 1 CPU to avoid issues with multiprocessing and easier debugging
    has_trace = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    has_breakpoint = sys.breakpointhook.__module__ != "sys"
    is_debug = has_trace or has_breakpoint
    if is_debug:
        print(f"Debug mode detected (has_trace={has_trace}, has_breakpoint={has_breakpoint}), using 1 CPU for easier debugging.")
        return 1
    # if running in GPULAB environment
    elif "GPULAB_CPUS_RESERVED" in os.environ:
        cpu_count = len(os.environ["GPULAB_CPUS_RESERVED"].split(","))
    # otherwise, use all available CPUs
    else: 
        cpu_count = os.cpu_count()
    cpu_count = max(0, cpu_count)
    print (f"Using {cpu_count} CPUs for data loading and processing (GPULAB_CPUS_RESERVED={os.environ.get('GPULAB_CPUS_RESERVED', 'N/A')}, total_cpus={cpu_count}).")
    return cpu_count

@dataclass
class TrainingConfig:
    """Configuration for training runs"""
    # Model class constructor and keyword arguments used to instantiate it.
    model_class: type = None
    model_args: dict = None

    # Maximum wall-clock training time in seconds.
    time_limit: int = 12 * 60 * 60
    
    # L2 regularization strength used by AdamW.
    weight_decay: float = 1e-2
    # Base learning rate used by the optimizer.
    learning_rate: float = 1e-4

    # Number of samples per optimization step.
    batch_size: int = 1024
    # Upper bound on full epochs.
    max_nr_epochs: int = 600
        
    # Early stopping patience (number of eval windows without improvement).
    patience: int = 5
    # Minimum change in validation loss to qualify as an improvement.
    minimum_delta: float = 0.01
    
    # Evaluate validation metrics every N epochs
    eval_every_n_epochs: int = 1
    # Log training metrics every N train batches.
    log_every_n_batches: int = 100

    compile_model: bool = True
    # Number of explicit warmup train steps after torch.compile. Set to 0 to disable.
    compile_warmup_steps: int = 5
 
    # Automatically retry training on CUDA OOM by lowering batch size.
    oom_retry_enabled: bool = True
    # Maximum number of CUDA OOM retries per run.
    oom_max_retries: int = 3
    # Smallest batch size allowed when retrying after CUDA OOM.
    min_batch_size: int = 8

    # Loss function factory/callable.
    criterion: Callable = CrossEntropyLoss
    # Optimizer class/factory.
    optimizer: Callable  = optim.AdamW
    # Optional scheduler factory/callable.
    scheduler: Callable = None


    # Fraction of each split to use when < 1.0.
    dataset_size: float = 1.0
    # Optional cap on validation batches per evaluation.
    validation_max_batches: int | None = None
    # Optional cap on test batches.
    test_max_batches: int | None = None
    # Callable that turns a raw dataloader batch into model inputs and labels.
    batch_preparation_fn: Callable | None = prepare_model_batch
    # Batch size used inside dataset.map for preprocessing.
    transform_batch_size:int = 1024
    # CPU workers used by dataset processing/DataLoader. Derived from GPULAB_CPUS_RESERVED when available.
    num_cpus:int = field(default_factory=calculate_num_cpus)

    

    def save(self, path):
        torch.save(asdict(self), os.path.join(path, "training_config.pt"))
        with open(os.path.join(path, "training_config.json"), "w") as f:
            # Make sure to convert any non-serializable fields to string if necessary
            d = asdict(self)
            d = {k: (str(v) if not isinstance(v, (int, float, str, dict, list, type(None))) else v) for k, v in d.items()}
            json.dump(d, f, indent=4)
    def reset_cpus(self):
        self.num_cpus = calculate_num_cpus()


    @classmethod
    def load(cls, json_path):
        from dataclasses import fields
        # print(f"Loading TrainingConfig from {json_path}...")
        print(f"Loading TrainingConfig from {json_path}...")
        # ensure the path has .json extension
        if not json_path.endswith(".json"):
            raise ValueError(f"Expected a .json file for loading TrainingConfig, got: {json_path}")
        
        with open(json_path, "r") as f:
            data = json.load(f)

        # Remove obsolete transforms.
        data.pop("training_transform", None)
        data.pop("validation_transform", None)

        # Convert strings such as "<class 'src.Models.SCNN.SCNN'>"
        # into importable Python classes.
        def resolve_class(value):
            if not isinstance(value, str) or not value.startswith("<class '"):
                return value

            module_and_class = value[len("<class '"):-2]
            module_name, class_name = module_and_class.rsplit(".", 1)

            # Migrate the old package name.
            module_name = module_name.replace(
                "src.Models", "src.models"
            )

            module = importlib.import_module(module_name)
            return getattr(module, class_name)

        for key in ("model_class", "criterion", "optimizer", "scheduler"):
            if key in data:
                data[key] = resolve_class(data[key])
        # Keep only fields supported by the current TrainingConfig.
        valid_fields = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)




@dataclass
class TestConfig():
    """Configuration for validation runs"""
    # Number of samples per batch during evaluation.
    batch_size: int = 32
    # Fraction of the dataset split to evaluate.
    dataset_size: float = 0.15
    # Loss function callable used for reporting evaluation loss.
    critrion: Callable = CrossEntropyLoss
    # Batch size used inside dataset.map during preprocessing.
    transform_batch_size:int = 32
    # Number of CPU workers used by the evaluation pipeline.
    num_cpus:int = field(default_factory=calculate_num_cpus)


