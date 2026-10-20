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
    eval_every_n_epochs=1,
    log_every_n_batches=100,
    batch_preparation_fn=prepare_model_batch,
    validation_max_batches=None,
    patience=10,
    minimum_delta=0.001,
    time_limit=None,
):
    # Set up device and AMP settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up paths for metrics and plots
    metric_path =os.path.join(model_folder, f'metrics_training_loop.pt')
    plot_path = os.path.join(model_folder, f'plots_training_loop.png')
    # Log initial configuration
    logging.info(f"Using device: {device}")
    logging.info(f"Using metrics path: {metric_path}")
    logging.info(f"Using patience: {patience}")
    # Log the lengths of the train and validation loaders, if available
    trainloader_len = len(trainloader) 
    validationloader_len = len(validationloader)
    logging.info(f"Training batches per pass: {trainloader_len}")
    logging.info(f"Validation batches per pass: {validationloader_len}")

    # Ensure that eval_every_n_epochs and log_every_n_batches are at least 1 to avoid division by zero or other issues.
    eval_every_n_epochs = max(1, eval_every_n_epochs)
    log_every_n_batches = max(1, log_every_n_batches)
    # Log the training configuration for clarity and reproducibility
    logging.info(f"Using eval_every_n_epochs: {eval_every_n_epochs}")
    logging.info(f"Using log_every_n_batches: {log_every_n_batches}")

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
    current_epoch = 0
    current_batch_index = 0
    train_iter = iter(trainloader)
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
    while current_epoch < max_epochs:
        model.train()
        data_wait_start = time.time()
        try:
            # Get next batch, or restart iterator if we've reached end of dataloader.
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
        prepared_inputs, prepared_labels = batch_preparation_fn(batch, device, scramble=True)
        interval_prep_time += time.time() - prep_start

        compute_start = time.time()
        batch_correct, batch_total, batch_loss = _train_single_batch_prepared_amp(
            model,
            prepared_inputs,
            prepared_labels,
            optimizer,
            criterion,
        )
        prepared_inputs = None
        prepared_labels = None

        interval_compute_time += time.time() - compute_start
        current_batch_index += 1

        interval_correct += batch_correct
        interval_total += batch_total
        interval_loss += batch_loss
        interval_batches += 1

        if current_batch_index % log_every_n_batches == 0:
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
                f"Batch {current_batch_index} | "
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
        else:
            batch = None  # Free batch memory
        should_eval = (current_epoch % eval_every_n_epochs == 0) or (current_epoch == max_epochs - 1)
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

            average_time_per_eval = (time.time() - start_time) / len(metrics.training_accuracy)
            interval_time = time.time() - interval_start

            logging.info(f'===============Eval checkpoint at step {current_batch_index} (epoch {current_epoch})===============')
            logging.info(f'\tTraining  \tAccuracy: {metrics.training_accuracy[-1]:.4f}%\tLoss: {metrics.train_loss[-1]:.4f}')
            logging.info(f'\tValidation\tAccuracy: {metrics.validation_accuracy[-1]:.4f}%\tLoss: {metrics.validation_loss[-1]:.4f}')
            logging.info('-' * 100)

            improved = metrics.model_improved()
            if improved:
                # remove old best model file if it exists
                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)
                    logging.info(f"\tRemoved old best model at {current_best_model_path}")
                current_best_model_path = save_model(model, model_folder, f'current_best_model_epoch_{current_epoch}')
                logging.info(f"\tSaved best model at epoch {current_epoch} to {current_best_model_path}")
                epochs_best_model = current_epoch
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
        
        current_epoch += 1

        if time_limit and (time.time() - start_time) > time_limit:
            logging.info("Time limit reached, stopping training.")
            break

    if best_model_state_dict is None:
        best_model_state_dict = model.state_dict()
        epochs_best_model = (current_epoch - 1) if current_epoch > 0 else 0

    logging.info('Finished Training')
    return best_model_state_dict, epochs_best_model,metrics



