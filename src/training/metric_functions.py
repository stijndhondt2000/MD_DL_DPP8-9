from time import perf_counter

import torch
from contextlib import nullcontext
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from src.infrastructure.logger import get_logger
from src.training.batch_preprocessing import prepare_model_batch
import numpy as np


logging = get_logger()

def _compute_label_prediction_statistics(labels, predictions, statistics_to_compute):
    stats = {}
    if len(labels) == 0:
        return stats

    if "accuracy" in statistics_to_compute:
        stats["accuracy"] = accuracy_score(labels, predictions)
    if "precision" in statistics_to_compute:
        stats["precision"] = precision_score(labels, predictions, average='weighted', zero_division=0)
    if "recall" in statistics_to_compute:
        stats["recall"] = recall_score(labels, predictions, average='weighted', zero_division=0)
    if "f1_score" in statistics_to_compute:
        stats["f1_score"] = f1_score(labels, predictions, average='weighted', zero_division=0)
    if "confusion_matrix" in statistics_to_compute:
        # make sure to specify labels to include all classes even if some are missing in this subset
        stats["confusion_matrix"] = confusion_matrix(labels, predictions, labels=list(range(5)))
        stats["confusion_matrix_normalized"] = confusion_matrix(labels, predictions, normalize='true', labels=list(range(5)))

    return stats

def calculate_fpr_multiclass(conf_matrix):
    """
    Calculate the False Positive Rate (FPR) for each class
    from a multiclass confusion matrix.

    Parameters
    ----------
    conf_matrix : np.ndarray
        NxN confusion matrix.

    Returns
    -------
    dict
        Dictionary mapping each class index to its FPR.
    """
    n_classes = conf_matrix.shape[0]

    if conf_matrix.shape[0] != conf_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    total = np.sum(conf_matrix)
    fpr = {}

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        tn = total - tp - fp - fn

        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    macro_fpr = np.mean(list(fpr.values()))

    return macro_fpr

def calculate_statistics(
    model,
    dataloader,
    criterion,
    device,
    max_batches: int | None = None,
    statistics_to_compute: list[str] | None = None,
    batch_preparation_fn=prepare_model_batch,
    split_name: str = "evaluation",
    log_every_n_batches: int = 100,
) -> dict[str, float]:
    """Calculate specified statistics for a model on a given dataloader. Loop over the dataloader and compute the specified statistics for each batch, then average them over the entire dataloader.
    Args:
        model (_type_): _model to evaluate
        dataloader (_type_): dataloader to evaluate on
        criterion (_type_): loss function to use for loss calculation
        device (_type_): device to use for evaluation
        max_batches (int | None, optional): maximum number of batches to evaluate. Defaults to None.
        statistics_to_compute (list[str] | None, optional): list of statistics to compute. Defaults to None.
        split_name: Label used in evaluation progress logs.
        log_every_n_batches: Number of batches between progress logs.

    Returns:
        dict[str, float]: dictionary containing the computed statistics
    """
    if statistics_to_compute is None:
        statistics_to_compute = ["accuracy", "loss", "precision", "recall", "f1_score", "confusion_matrix"]
    batch_preparation_fn = batch_preparation_fn or prepare_model_batch
    
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    num_batches = 0
    log_every_n_batches = max(1, log_every_n_batches)
    interval_batches = interval_frames = interval_correct = 0
    interval_loss = interval_wait = interval_prep = interval_compute = 0.0
    dpp_class_examples = {
        "dpp8": {"labels": [], "predictions": []},
        "dpp9": {"labels": [], "predictions": []},
    }
    has_dpp_class = False

    model.to(device)
    model.eval()

    with torch.no_grad():
        batches = iter(dataloader)
        while max_batches is None or num_batches < max_batches:
            wait_start = perf_counter()
            try:
                data = next(batches)
            except StopIteration:
                break
            prep_start = perf_counter()
            interval_wait += prep_start - wait_start
            images, labels = batch_preparation_fn(data, device, scramble=False)
            compute_start = perf_counter()
            interval_prep += compute_start - prep_start

            outputs = model(images)
            loss = criterion(outputs, labels).item()
            total_loss += loss
            num_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()

            all_labels.extend(labels_cpu)
            all_predictions.extend(predicted_cpu)

            interval_correct += np.count_nonzero(labels_cpu == predicted_cpu)
            interval_frames += len(labels_cpu)
            interval_loss += loss
            interval_batches += 1
            interval_compute += perf_counter() - compute_start

            if num_batches == 1 or num_batches % log_every_n_batches == 0:
                interval_time = max(interval_wait + interval_prep + interval_compute, 1e-9)
                logging.info(
                    "%s | Batch %d | Frames: %d | Eval Acc: %.4f%% | Eval Loss: %.4f | "
                    "Avg Data Wait: %.1fms | Avg Prep: %.1fms | Avg Compute: %.1fms | "
                    "Samples/s: %.1f | Shares(wait/prep/compute): %.1f%%/%.1f%%/%.1f%%",
                    split_name, num_batches, len(all_labels),
                    100 * interval_correct / max(interval_frames, 1),
                    interval_loss / interval_batches,
                    1000 * interval_wait / interval_batches,
                    1000 * interval_prep / interval_batches,
                    1000 * interval_compute / interval_batches,
                    interval_frames / interval_time,
                    100 * interval_wait / interval_time,
                    100 * interval_prep / interval_time,
                    100 * interval_compute / interval_time,
                )
                interval_batches = interval_frames = interval_correct = 0
                interval_loss = interval_wait = interval_prep = interval_compute = 0.0

    # statistics -> dpp_class -> accuracy, loss, precision, recall, f1_score, confusion_matrix    
    statistics = {}

    
    if "accuracy" in statistics_to_compute:
        statistics["accuracy"] = accuracy_score(all_labels, all_predictions)
    if "loss" in statistics_to_compute:
        statistics["loss"] = total_loss / max(num_batches, 1)
    if "precision" in statistics_to_compute:
        statistics["precision"] = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "recall" in statistics_to_compute:
        statistics["recall"] = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "f1_score" in statistics_to_compute:
        statistics["f1_score"] = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "confusion_matrix" in statistics_to_compute:
        statistics["confusion_matrix"] = confusion_matrix(all_labels, all_predictions, labels=list(range(5)))
        statistics["confusion_matrix_normalized"] = confusion_matrix(all_labels, all_predictions, normalize='true', labels=list(range(5)))
    if "fpr" in statistics_to_compute:
        statistics["fpr"] = calculate_fpr_multiclass(statistics["confusion_matrix"])

    logging.info(
        "%s evaluation complete | Batches: %d | Frames: %d",
        split_name, num_batches, len(all_labels),
    )
    return statistics

def calculate_accuracy_and_loss(
    model,
    dataloader,
    criterion,
    device,
    max_batches: int | None = None,
    batch_preparation_fn=prepare_model_batch,
):
    correct = 0
    total = 0
    current_loss = 0.0
    num_batches = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        # iterate over streaming dataloader
        for batch_idx, data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = batch_preparation_fn(data, device, scramble=False)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_loss += loss.item()
            num_batches += 1

    accuracy = correct / max(total, 1)
    current_loss /= max(num_batches, 1)

    return accuracy, current_loss

def calculate_precision_recall_f1(
    model,
    dataloader,
    device,
    max_batches: int | None = None,
    batch_preparation_fn=prepare_model_batch,
):
    all_labels = []
    all_predictions = []
    batch_preparation_fn = batch_preparation_fn or prepare_model_batch
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = batch_preparation_fn(data, device, scramble=False)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions,normalize='true', labels=list(range(5)))

    return precision, recall, f1, conf_matrix

def all_statistics(
    model,
    dataloader,
    criterion,
    device,
    max_batches: int | None = None,
    batch_preparation_fn=prepare_model_batch,
) -> dict[str, float]:
    accuracy, loss = calculate_accuracy_and_loss(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
        batch_preparation_fn=batch_preparation_fn,
    )
    precision, recall, f1, conf_matrix = calculate_precision_recall_f1(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        batch_preparation_fn=batch_preparation_fn,
    )

    return {
        "accuracy": accuracy,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }
