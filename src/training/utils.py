"""Shared training, checkpoint, split, and CUDA prefetch utilities."""

import os
import time
import threading
from contextlib import nullcontext
import datasets
from src.data.data_preprocessing.trajectory_split import split_trajectories

import torch
from src.training.Metrics import Metrics
from src.training.batch_preprocessing import prepare_model_batch
from src.training.metric_functions import calculate_accuracy_and_loss
from sklearn.metrics import confusion_matrix
from src.utils.resolvers import (
    _load_weights,
)

from src.infrastructure.logger import get_logger
logging = get_logger() 
def save_model(model, model_folder, model_name):
    """Serialize a complete PyTorch model under ``model_folder``."""
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f"{model_name}.pth")
    torch.save(model, model_path)
    logging.info(f"Saved full model to {model_path}")
    return model_path



def load_model(model_filepath, device=None, model_class=None, model_args=None):
    """Load a full model or reconstruct one from a state-dict checkpoint.

    If ``model_class`` is omitted, dictionary checkpoints are returned unchanged.
    Reconstructed and full models are moved to ``device`` and set to evaluation
    mode.
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(model_filepath, map_location=device, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        checkpoint.to(device)
        checkpoint.eval()
        return checkpoint

    if isinstance(checkpoint, dict):
        if model_class is None:
            return checkpoint

        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("model_state_dict", checkpoint)

        model = model_class(**(model_args or {}))
        _load_weights(model, state_dict)
        model.to(device)
        model.eval()
        return model

    raise TypeError(
        f"Unsupported checkpoint type {type(checkpoint)!r} in {model_filepath}. Expected a torch.nn.Module or state dict."
    )

def model_name(model_name_prefix):
    """Create a timestamped model directory name from a readable prefix."""
    date = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{model_name_prefix}_{date}"
    return model_name
def get_device():
    """Select CUDA when available, otherwise return CPU with a warning."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: Training on CPU, this may be slow. Consider using a GPU for faster training.")
    return device

def get_subset(dataset, fraction, shuffle=True, seed=42):
    """Return a deterministic fraction of a dataset, optionally shuffled."""
    if 0.9999 < fraction <= 1.0:
        return dataset
    if fraction <= 0 or fraction > 1.0001:
        raise ValueError("Fraction must be between 0 and 1.")
    dataset_size = int(max(len(dataset) * fraction, 1))
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(dataset_size))
def clear_cache(dataset_dir):
    """Remove Hugging Face dataset cache files from a disk-backed dataset."""
    ds = datasets.load_from_disk(dataset_dir)
    ds.cleanup_cache_files()
    print(f"Cleared cache files in dataset at {dataset_dir}")



def train_val_test_split(dataset, train_fraction=0.7, val_fraction=0.15, seed=42):
    """Split whole trajectories; retain each molecular condition in every set."""
    splits, _ = split_trajectories(dataset, train_fraction, val_fraction, seed)
    return splits

def training_phase(model, trainloader, optimizer, criterion, device):
    """Train over a finite dataloader and return aggregate accuracy/loss."""
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0
    for i, batch in enumerate(trainloader, 0):
        inputs, labels = prepare_model_batch(batch, device, scramble=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()
        num_batches += 1

    train_loss = running_loss / max(num_batches, 1)
    train_acc = correct / max(total, 1)
    return train_acc, train_loss


def _train_single_batch(model, batch, optimizer, criterion, device):
    """Train one batch and return correct predictions, sample count and loss."""
    model.train()
    inputs, labels = prepare_model_batch(batch, device, scramble=True)

    return _train_single_batch_prepared(model, inputs, labels, optimizer, criterion)


def _train_single_batch_prepared(model, inputs, labels, optimizer, criterion):
    """Train one already-prepared batch and return correct predictions, sample count and loss."""
    model.train()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    batch_total = labels.size(0)
    batch_correct = (predicted == labels).sum().item()
    return batch_correct, batch_total, loss.item()


def _train_single_batch_prepared_amp(
    model,
    inputs,
    labels,
    optimizer,
    criterion,
):
    """Train one already-prepared batch with optional CUDA AMP and return stats."""
    model.train()
    optimizer.zero_grad()


    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, predicted = torch.max(outputs, 1)
    batch_total = labels.size(0)
    batch_correct = (predicted == labels).sum().item()
    return batch_correct, batch_total, loss.item()



def _safe_len(dataloader):
    """Return dataloader length when available, otherwise None."""
    try:
        return len(dataloader)
    except (TypeError, AttributeError):
        return None

