import os
import gc
from datetime import datetime
import torch
from dotenv import load_dotenv
from src.data.data_loading.HFDataloader import initialize_dataloaders
from src.training.model_testing import model_testing
from src.training.utils import get_device
from src.training.training_loop import training_loop
from src.infrastructure.dotenvManager import dotenvManager
from src.config.training_config import TrainingConfig, TestConfig
from src.training.training_loop import training_loop
import torch.nn as nn
import os
from src.infrastructure.logger import init_logger, get_logger, DEBUG as LOGGING_DEBUG
logging = get_logger() 

def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True when exception indicates a CUDA OOM condition."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "cudnn_status_alloc_failed" in msg


def _cleanup_cuda_memory():
    """Release cached CUDA memory between retries."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass





def save_results(model_state_dict, model_dir: str, model_name: str, metrics):
    """Save model, metrics, and test accuracy"""
    filepath = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model_state_dict, filepath)
    logging.info(f"Model saved to: {filepath}")

    plot_path = filepath.replace('.pth', '.png')
    metric_path = filepath.replace('.pth', '.metrics')
    metrics.save_plot("Model Performance", plot_path)
    metrics.save_metrics(metric_path)

    logging.info(f"Plot saved to: {plot_path}")


def _warmup_with_batch_fn(model, dataloader, optimizer, criterion, device, batch_preparation_fn, steps=5):
    train_iter = iter(dataloader)
    for step_idx in range(steps):
        fetch_start = datetime.now()
        try:
            batch = next(train_iter)
        except StopIteration:
            break
        fetch_elapsed = (datetime.now() - fetch_start).total_seconds()
        step_start = datetime.now()
        model.train()
        inputs, labels = batch_preparation_fn(batch, device, scramble=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        step_elapsed = (datetime.now() - step_start).total_seconds()
        logging.info(f"Warmup step {step_idx + 1}/{steps}: fetch={fetch_elapsed:.2f}s, train_step={step_elapsed:.2f}s")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def initialize_run_directory(model_name):
    """Create model directory and setup logging"""
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.getenv("OUTPUT_DIR")
    model_dir = os.path.join(output_dir, "models", f"{model_name}", time_string)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir
    

def train_model(config: TrainingConfig, model_name: str):
    """Main training function - single entry point for all models"""
    load_dotenv() # Load environment variables from .env file
    batch_preparation_fn = config.batch_preparation_fn 
    
    # Setup
    run_dir = initialize_run_directory(model_name)
    logger = init_logger(model_dir=run_dir, log_mode=LOGGING_DEBUG, log_file="debug.log")
    logger.info(f"Starting training for {model_name} with, saving to {run_dir}")
    logger.info(f"Training configuration: {config}")
    logger.info(f"Weight Decay: {config.weight_decay}, Learning Rate: {config.learning_rate}")
    logger.info(f"Max Num Epochs: {config.max_nr_epochs}")
    # Initialize the cache manager
    dotenv_manager = dotenvManager()
    # Determine the maximum number of retries for OOM errors
    max_retries = max(0, int(config.oom_max_retries)) if config.oom_retry_enabled else 0
    attempt = 0
    training_result = None
    try:
        while True:
            model = None
            dataloaders = None
            try:
                # Save the config for this run
                config.save(os.path.join(run_dir))

                # Initialize model and data for this attempt (batch size may change after OOM).
                model = config.model_class(**config.model_args)
                dataloaders = initialize_dataloaders(config, dotenv_manager=dotenv_manager)

                # Setup training
                device = get_device()
                criterion = config.criterion()
                if isinstance(criterion, nn.Module):
                    criterion = criterion.to(device)
                optimizer = config.optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                
                

                # Set up device and CUDA settings before moving model to device
                if device.type == "cuda":
                    # Throughput-oriented CUDA backend settings.
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.set_float32_matmul_precision("high")

                # Move model to device and compile if requested
                model.to(device)
                if config.compile_model:
                    logger.info("Compiling model with torch.compile() for potentially faster training.")
                    model = torch.compile(model)
                    warmup_steps = max(0, int(getattr(config, "compile_warmup_steps", 1)))
                    if warmup_steps > 0:
                        logger.info(f"Warming up compiled model for {warmup_steps} step(s)...")
                        _warmup_with_batch_fn(
                            model,
                            dataloaders["train"],
                            optimizer,
                            criterion,
                            device,
                            batch_preparation_fn=batch_preparation_fn,
                            steps=warmup_steps,
                        )
                    else:
                        logger.info("Skipping explicit compile warmup (compile_warmup_steps=0).")

                # Train the model
                model_state_dict, nr_epochs, metrics = training_loop(
                    model=model,
                    model_folder=run_dir,
                    trainloader=dataloaders["train"],
                    validationloader=dataloaders["validation"],
                    optimizer=optimizer,
                    criterion=criterion,
                    max_epochs=config.max_nr_epochs,
                    eval_every_n_epochs=config.eval_every_n_epochs,
                    log_every_n_batches=config.log_every_n_batches,
                    batch_preparation_fn=batch_preparation_fn,
                    validation_max_batches=config.validation_max_batches,
                    patience=config.patience,
                    minimum_delta=config.minimum_delta,
                    time_limit=config.time_limit
                )
                # After training, restore the model state to the best checkpoint for testing and saving results
                model.load_state_dict(model_state_dict)
                # Run testing on the best model
                model_testing(
                    model,
                    dataloaders["test"],
                    criterion,
                    device,
                    batch_preparation_fn=batch_preparation_fn,
                    output_dir=run_dir,
                    max_batches=config.test_max_batches,
                )
                # Save the model and metrics after training and testing
                save_results(model_state_dict, run_dir, model_name, metrics)
                logger.info("Training complete.")
                # Return the training result with relevant metrics and configuration
                best_validation_accuracy = max(metrics.validation_accuracy) if metrics.validation_accuracy else None
                best_validation_loss = min(metrics.validation_loss) if metrics.validation_loss else None
                # Return a dictionary containing the training results, including run directory, number of epochs, best validation metrics, final validation metrics, and the training configuration.
                training_result = {
                    "run_dir": run_dir,
                    "nr_epochs": nr_epochs,
                    "best_validation_accuracy": best_validation_accuracy,
                    "best_validation_loss": best_validation_loss,
                    "final_validation_accuracy": metrics.validation_accuracy[-1] if metrics.validation_accuracy else None,
                    "final_validation_loss": metrics.validation_loss[-1] if metrics.validation_loss else None,
                    "config": config,
                }
                break
            # Handle CUDA OOM errors and retry with reduced batch size if configured to do so 
            except Exception as exc:
                is_oom = _is_cuda_oom_error(exc)
                can_retry = (
                    is_oom
                    and torch.cuda.is_available()
                    and config.oom_retry_enabled
                    and attempt < max_retries
                    and config.batch_size > max(1, int(config.min_batch_size))
                )
                #  If the exception is not a CUDA OOM error or if retries are exhausted, re-raise the exception to propagate it up the call stack.
                if not can_retry:
                    raise
                # If a CUDA OOM error occurred and retries are allowed, reduce the batch size and retry training.
                old_batch_size = int(config.batch_size)
                new_batch_size = max(int(config.min_batch_size), old_batch_size // 2)
                # If the new batch size is not smaller than the old batch size, raise an error to prevent an infinite loop of retries.
                if new_batch_size >= old_batch_size:
                    raise
                # Log the OOM error and the new batch size, update the configuration, and clean up CUDA memory before retrying.
                attempt += 1
                logger.info(
                    f"CUDA OOM detected (attempt {attempt}/{max_retries}). "
                    f"Reducing batch size from {old_batch_size} to {new_batch_size} and retrying..."
                )
                # Update the configuration with the new batch size for the next training attempt.
                config.batch_size = new_batch_size
                # Clean up CUDA memory before retrying to free up resources for the next attempt.
                _cleanup_cuda_memory()
            except SystemExit as e:
                # Catch and log unexpected SystemErrors that may occur during training (e.g., from torch.compile internals).
                logger.info(f"Job halted with exit code {e.code}")
                # Re-raise to allow external handlers (e.g., job schedulers) to detect the exit condition.
                raise
            finally:
                # Ensure that model and dataloaders are deleted to free up memory, and perform CUDA memory cleanup.
                if model is not None:
                    del model
                if dataloaders is not None:
                    del dataloaders
                _cleanup_cuda_memory()
    except Exception as final_exc:
        # If training fails after all retries, log the failure and raise the final exception to propagate it up the call stack.
        logger.info(f"Training failed after {attempt} attempt(s) with batch size {config.batch_size}.")
        raise final_exc
    # Return the training result dictionary containing relevant metrics and configuration for further analysis or reporting.
    return training_result

