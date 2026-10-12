import os
import time
import torch
from src.training.Metrics import Metrics
from src.training.batch_preprocessing import prepare_model_batch
from src.training.metric_functions import calculate_accuracy_and_loss

from src.training.utils import (
    _train_single_batch_prepared_amp,
    _safe_len,
    save_model,
    _resolve_amp_dtype,
    CUDABatchPrefetcher,
)
from src.infrastructure.logger import get_logger
logging = get_logger() 

def training_loop(
    model,
    trainloader,
    validationloader,
    optimizer,
    criterion,
    model_folder,
    scheduler=None,
    max_epochs=200,
    max_train_steps=None,
    steps_per_epoch=1000,
    eval_every_steps=1000,
    log_every_steps=100,
    use_cuda_prefetcher=True,
    batch_preparation_fn=prepare_model_batch,
    use_mixed_precision=False,
    amp_dtype="bfloat16",
    validation_max_batches=None,
    patience=10,
    minimum_delta=0.001,
    time_limit=None,
):
    # Set up device and AMP settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    use_amp = bool(use_mixed_precision and device.type == "cuda")
    use_grad_scaler = bool(use_amp and resolved_amp_dtype == torch.float16)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    # Set up paths for metrics and plots
    metric_path =os.path.join(model_folder, f'metrics_training_loop.pt')
    plot_path = os.path.join(model_folder, f'plots_training_loop.png')
    # Log initial configuration
    logging.info(f"Using device: {device}")
    logging.info(f"Using metrics path: {metric_path}")
    logging.info(f"Using patience: {patience}")
    # Log the lengths of the train and validation loaders, if available
    trainloader_len = _safe_len(trainloader)
    validationloader_len = _safe_len(validationloader)
    if trainloader_len is not None:
        logging.info(f"Training batches per pass: {trainloader_len}")
    else:
        logging.info("Training loader is streaming/iterable (unknown length per pass).")

    if validationloader_len is not None:
        logging.info(f"Validation batches per pass: {validationloader_len}")
    else:
        logging.info("Validation loader length is unknown.")

    # Determine max_train_steps based on max_epochs and trainloader length if not explicitly set
    if max_train_steps is None:
        if trainloader_len is not None and max_epochs is not None:
            max_train_steps = max_epochs * trainloader_len
            steps_per_epoch = trainloader_len
            eval_every_steps = trainloader_len
        else:
            max_train_steps = max_epochs * steps_per_epoch
    # Ensure that eval_every_steps, log_every_steps, and steps_per_epoch are at least 1 to avoid division by zero or other issues.
    eval_every_steps = max(1, eval_every_steps)
    log_every_steps = max(1, log_every_steps)
    steps_per_epoch = max(1, steps_per_epoch)
    # Log the training configuration for clarity and reproducibility
    logging.info(f"Using max_train_steps: {max_train_steps}")
    logging.info(f"Using steps_per_epoch: {steps_per_epoch}")
    logging.info(f"Using eval_every_steps: {eval_every_steps}")
    logging.info(f"Using log_every_steps: {log_every_steps}")

    # time limit in readable format for logging
    if time_limit:
        time_limit_str = f"{time_limit // 3600}h {(time_limit % 3600) // 60}m {time_limit % 60}s"
        logging.info(f"Using time limit: {time_limit_str}.")
    logging.info(f"Using plot path: {plot_path}")

    if device == torch.device("cpu"):
        logging.warning("Training on CPU, this may be slow. Consider using a GPU for faster training.")


    best_model_state_dict = None
    epochs_best_model = None
    metrics = Metrics(patience=patience, minimum_delta=minimum_delta)

    start_time = time.time()

    global_step = 0
    train_iter = iter(trainloader)
    prefetcher = None
    if device.type == "cuda" and use_cuda_prefetcher:
        logging.info("CUDA prefetcher enabled: overlapping next-batch preprocessing with current compute.")
        prefetcher = CUDABatchPrefetcher(
            trainloader,
            device=device,
            scramble=True,
            batch_preparation_fn=batch_preparation_fn,
        )
        prefetcher.reset()
    interval_correct = 0
    interval_total = 0
    interval_loss = 0.0
    interval_batches = 0
    interval_data_wait = 0.0
    interval_prep_time = 0.0
    interval_compute_time = 0.0
    interval_start = time.time()
    logging.info("Starting training loop...")
    current_best_model_path = None
    amp_fallback_applied = False
    while global_step < max_train_steps:
        model.train()
        data_wait_start = time.time()
        try:
            if prefetcher is not None:
                inputs, labels = prefetcher.next()
            else:
                # Get next batch, or restart iterator if we've reached end of dataloader.
                batch = next(train_iter)
        except StopIteration:
            if prefetcher is not None:
                prefetcher.reset()
                inputs, labels = prefetcher.next()
            else:
                train_iter = iter(trainloader)
                batch = next(train_iter)
        except RuntimeError as exc:
            if "DataLoader worker" in str(exc):
                raise RuntimeError(
                    "Training dataloader worker crashed. This is often caused by too many workers "
                    "or worker-side exceptions in transforms. Try lowering TrainingConfig.num_cpus "
                    "(for streamed datasets start with 0-2 workers)."
                ) from exc
            raise

        interval_data_wait += time.time() - data_wait_start

        prep_start = time.time()
        if prefetcher is not None:
            # In prefetch mode, prep reflects stream sync + handoff cost.
            interval_prep_time += time.time() - prep_start
            compute_start = time.time()
            batch_correct, batch_total, batch_loss = _train_single_batch_prepared_amp(
                model,
                inputs,
                labels,
                optimizer,
                criterion,
                device,
                use_mixed_precision=use_amp,
                amp_dtype=resolved_amp_dtype,
                grad_scaler=grad_scaler,
            )
        else:
            prepared_inputs, prepared_labels = batch_preparation_fn(batch, device, scramble=True)
            interval_prep_time += time.time() - prep_start

            compute_start = time.time()
            batch_correct, batch_total, batch_loss = _train_single_batch_prepared_amp(
                model,
                prepared_inputs,
                prepared_labels,
                optimizer,
                criterion,
                device,
                use_mixed_precision=use_amp,
                amp_dtype=resolved_amp_dtype,
                grad_scaler=grad_scaler,
            )
            prepared_inputs = None
            prepared_labels = None

        if not torch.isfinite(torch.tensor(batch_loss)):
            if use_amp and not amp_fallback_applied:
                logging.warning(
                    "Non-finite loss detected with AMP; disabling mixed precision for the rest of this run."
                )
                use_amp = False
                grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
                amp_fallback_applied = True
                continue

            logging.warning("Skipping non-finite batch loss.")
            continue

        interval_compute_time += time.time() - compute_start
        global_step += 1

        interval_correct += batch_correct
        interval_total += batch_total
        interval_loss += batch_loss
        interval_batches += 1

        if global_step % log_every_steps == 0:
            running_acc = interval_correct / max(interval_total, 1)
            running_loss = interval_loss / max(interval_batches, 1)
            avg_data_wait_ms = (interval_data_wait / max(interval_batches, 1)) * 1000
            avg_prep_ms = (interval_prep_time / max(interval_batches, 1)) * 1000
            avg_compute_ms = (interval_compute_time / max(interval_batches, 1)) * 1000
            total_interval_time = max(interval_data_wait + interval_prep_time + interval_compute_time, 1e-9)
            samples_per_sec = interval_total / total_interval_time
            data_wait_fraction = interval_data_wait / total_interval_time
            prep_fraction = interval_prep_time / total_interval_time
            compute_fraction = interval_compute_time / total_interval_time
            logging.info(
                f"Step {global_step}/{max_train_steps} | "
                f"Train Acc: {running_acc * 100:.4f}% | "
                f"Train Loss: {running_loss:.4f} | "
                f"Avg Data Wait: {avg_data_wait_ms:.1f}ms | "
                f"Avg Prep: {avg_prep_ms:.1f}ms | "
                f"Avg Compute: {avg_compute_ms:.1f}ms | "
                f"Samples/s: {samples_per_sec:.1f} | "
                f"Shares(wait/prep/compute): "
                f"{data_wait_fraction * 100:.1f}%/"
                f"{prep_fraction * 100:.1f}%/"
                f"{compute_fraction * 100:.1f}%"
            )
        if prefetcher is not None:
            inputs = None
            labels = None
        else:
            batch = None  # Free batch memory
        should_eval = (global_step % eval_every_steps == 0) or (global_step == max_train_steps)
        if should_eval:
            model.eval()
            

            train_acc = interval_correct / max(interval_total, 1)
            train_loss = interval_loss / max(interval_batches, 1)
            try:
                val_acc, val_loss = calculate_accuracy_and_loss(
                    model,
                    validationloader,
                    criterion,
                    device,
                    max_batches=validation_max_batches,
                    batch_preparation_fn=batch_preparation_fn,
                    use_mixed_precision=use_amp,
                    amp_dtype=resolved_amp_dtype,
                )
            except RuntimeError as exc:
                if "DataLoader worker" in str(exc):
                    raise RuntimeError(
                        "Validation dataloader worker crashed at evaluation. "
                        "Reduce TrainingConfig.num_cpus and/or set validation workers to 0 for streamed data."
                    ) from exc
                raise
            if scheduler:
                scheduler.step()

            metrics.update(train_acc * 100, train_loss, val_acc * 100, val_loss)
            metrics.save_metrics(metric_path)
            metrics.save_plot("Training and Validation Metrics", plot_path)

            pseudo_epoch = (global_step - 1) // steps_per_epoch + 1
            average_time_per_eval = (time.time() - start_time) / len(metrics.training_accuracy)
            interval_time = time.time() - interval_start

            logging.info(f'===============Eval checkpoint at step {global_step} (pseudo-epoch {pseudo_epoch})===============')
            logging.info(f'\tTraining  \tAccuracy: {metrics.training_accuracy[-1]:.4f}%\tLoss: {metrics.train_loss[-1]:.4f}')
            logging.info(f'\tValidation\tAccuracy: {metrics.validation_accuracy[-1]:.4f}%\tLoss: {metrics.validation_loss[-1]:.4f}')
            logging.info('-' * 100)

            improved = metrics.model_improved()
            if improved:
                # remove old best model file if it exists
                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)
                    logging.info(f"\tRemoved old best model at {current_best_model_path}")
                current_best_model_path = save_model(model, model_folder, f'current_best_model_epoch_{pseudo_epoch}.pth')
                logging.info(f"\tSaved best model at step {global_step} to {current_best_model_path}")
                epochs_best_model = pseudo_epoch
            else:
                metrics.patience_counter += 1

            logging.info(f'\tPatience Counter: {metrics.patience_counter}/{patience}')
            logging.info(
                f'\tTime for eval interval: {interval_time // 60:.2f}m {interval_time % 60:.0f}s\t'
                f'(avg: {average_time_per_eval // 60}m {average_time_per_eval % 60:.0f}s/check)\t'
                f'Time elapsed since start: {(time.time() - start_time) // 60:.0f}m'
            )
            logging.info(
                f'\tPipeline profile: data_wait={interval_data_wait:.2f}s, '
                f'prep={interval_prep_time:.2f}s, '
                f'compute={interval_compute_time:.2f}s, '
                f'data_wait_share={100 * interval_data_wait / max(interval_data_wait + interval_prep_time + interval_compute_time, 1e-9):.1f}%'
            )
            if scheduler:
                logging.info(f'\tLearning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            logging.info('=' * 100)

            interval_correct = 0
            interval_total = 0
            interval_loss = 0.0
            interval_batches = 0
            interval_data_wait = 0.0
            interval_prep_time = 0.0
            interval_compute_time = 0.0
            interval_start = time.time()

            if metrics.patience_counter >= patience:
                logging.info("Early stopping due to overfitting.")
                break

        if time_limit and (time.time() - start_time) > time_limit:
            logging.info("Time limit reached, stopping training.")
            break

    if best_model_state_dict is None:
        best_model_state_dict = model.state_dict()
        epochs_best_model = (global_step - 1) // steps_per_epoch + 1 if global_step > 0 else 0

    logging.info('Finished Training')
    return best_model_state_dict, epochs_best_model,metrics



