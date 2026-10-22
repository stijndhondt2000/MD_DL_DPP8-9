from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import Dataset, NamedSplit, concatenate_datasets, load_from_disk
import os
from pathlib import Path


from src.training.utils import train_val_test_split
from src.utils.molecular_parsing import parse_filename
from src.data.data_preprocessing.utils import extract_coordinates, parse_pdb_streaming
from src.infrastructure.dotenvManager import dotenvManager
import shutil


def _process_tar_file_to_dataset(tar_file: Path, streaming_pdb_dataset_path: str, skip_existing: bool = True, min_expected_size: int = 3.5 * 10**9):
    """Build one dataset shard for a single TAR archive."""
    filename = tar_file.stem
    print(f"Processing file {filename}...")

    dpp_class, ligand_name, binding_type, replica_id = parse_filename(filename)
    split_name = f"{dpp_class}_{binding_type}_{ligand_name}_{replica_id}"
    res_dir = os.path.join(streaming_pdb_dataset_path, split_name)

    if os.path.exists(res_dir):
        existing_size = 0
        for element in os.scandir(res_dir):
            if element.is_file():
                existing_size += os.path.getsize(element)
        if skip_existing and existing_size >= min_expected_size:
            print(f"✓ Skipping {filename} as it already exists and is of expected size.")
            return {"status": "skipped", "path": res_dir, "filename": filename}

    dataset = Dataset.from_generator(
        parse_pdb_streaming,
        gen_kwargs={
            "tar_path": str(tar_file),
            "dpp_class": dpp_class,
            "ligand_name": ligand_name,
            "binding_type": binding_type,
            "replica_id": replica_id,
        },
        split=NamedSplit(split_name),
    )

    os.makedirs(res_dir, exist_ok=True)
    dataset.save_to_disk(res_dir, max_shard_size="4GB")
    print(f"✓ Processed {filename} and saved to {res_dir}.")
    return {"status": "processed", "path": res_dir, "filename": filename}


def generate_dataset_from_tars(streaming_pdb_dataset_path, tar_folder, num_proc=32, skip_existing=True):
    """Convert every TAR file in ``tar_folder`` into a disk-backed dataset shard.

    The output directory ends up with one subdirectory per trajectory. Each
    subdirectory contains a Hugging Face dataset saved with ``save_to_disk`` so
    the full corpus can later be reloaded and concatenated without reparsing the
    source TAR archives.
    """

    print(f"Creating dataset {streaming_pdb_dataset_path} from TAR files {tar_folder}")
    tar_files = sorted(Path(tar_folder).glob('*.tar.gz'))
    if not tar_files:
        raise FileNotFoundError(f"No TAR.gz files found in {tar_folder}. Please ensure the TAR files are present.")

    min_expected_size = 3.5 * 10**9 # 3.5 GB

    if num_proc == 1:
        for tar_file in tar_files:
            _process_tar_file_to_dataset(
                tar_file=tar_file,
                streaming_pdb_dataset_path=streaming_pdb_dataset_path,
                skip_existing=skip_existing,
                min_expected_size=min_expected_size,
            )
    else:
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            futures = {
                executor.submit(
                    _process_tar_file_to_dataset,
                    tar_file,
                    streaming_pdb_dataset_path,
                    skip_existing,
                    min_expected_size,
                ): tar_file
                for tar_file in tar_files
            }
            for future in as_completed(futures):
                tar_file = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Failed to process {tar_file}: {exc}")
                    raise

    print("✓ Streaming dataset loaded and saved to disk.")

def generate_external_dataset_from_tars(regenerate=False, num_proc=32, skip_existing=True, remove_temp_dataset=True):
    """Build the external dataset from all TAR shards.

    The function first materializes per-trajectory datasets on disk, then reloads
    them, concatenates them into one large dataset, shuffles the rows, and finally
    applies the project-specific train/val/test split.
    """
    dotenv_manager = dotenvManager()
    # folder with tar files
    tar_folder = dotenv_manager.get_compressed_external_dataset_directory()
    # ensure tar foler exists and is not empty
    if not os.path.exists(tar_folder) or len(os.listdir(tar_folder)) == 0:
        raise FileNotFoundError(f"TAR folder {tar_folder} does not exist or is empty. Please ensure the TAR files are present.")
    # folder to store the intermediate dataset cache
    print("Starting external data preprocessing...")
    dataset_path = dotenv_manager.get_dataset_directory()
    temp_dataset_path = os.path.join(dataset_path, "temp", "external")
    external_dataset_path = os.path.join(dataset_path, "external")
    
    # Ensure the intermediate dataset cache exists before writing any shards.
    os.makedirs(temp_dataset_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(external_dataset_path, exist_ok=True)

    if not regenerate and os.path.exists(external_dataset_path):
        if "dataset_info.json" in os.listdir(external_dataset_path):
            print(f"External dataset already exists at {external_dataset_path}. Skipping regeneration as regenerate is False.")
            return

    # Step 1: turn each TAR archive into a saved dataset shard on disk.
    generate_dataset_from_tars(temp_dataset_path, tar_folder, num_proc=num_proc, skip_existing=skip_existing)

    full_ds = []
    # Step 2: reload every shard and keep the partial datasets in memory long
    # enough to concatenate them into a single dataset object.
    for last_child in os.listdir(temp_dataset_path):
        traj_path = os.path.join(temp_dataset_path, last_child)

        print(f"Loading dataset for traj: {last_child}...")
        partial_ds = load_from_disk(traj_path)
        full_ds.append(partial_ds)

    # Merge all trajectory-level shards into one dataset before shuffling and
    # splitting. This gives the training pipeline a single consistent dataset view.
    full_ds = concatenate_datasets(full_ds)
    print(f"✓ Full dataset loaded with {len(full_ds)} samples.")

    # Log the final split size so preprocessing output can be verified quickly.
    print(f"\tFinal external dataset size: {len(full_ds)}")

    print("Saving final external dataset to disk...")

    full_ds.save_to_disk(external_dataset_path, max_shard_size="4GB", num_proc=num_proc)

    # remove the temporary dataset cache to free up disk space after the final splits are saved.
    if os.path.exists(temp_dataset_path) and remove_temp_dataset:
        shutil.rmtree(temp_dataset_path)
        print(f"Removed temporary dataset cache at {temp_dataset_path}.")
    print("Datasets saved.")


def generate_development_dataset_from_tars(regenerate=False, num_proc=32, skip_existing=True, remove_temp_dataset=True):
    """Build the final train/validation/test dataset from all TAR shards.

    The function first materializes per-trajectory datasets on disk, then reloads
    them, concatenates them into one large dataset, shuffles the rows, and finally
    applies the project-specific train/val/test split.
    """
    dotenv_manager = dotenvManager()
    # folder with tar files
    tar_folder = dotenv_manager.get_compressed_dataset_directory()
    # ensure tar foler exists and is not empty
    if not os.path.exists(tar_folder) or len(os.listdir(tar_folder)) == 0:
        raise FileNotFoundError(f"TAR folder {tar_folder} does not exist or is empty. Please ensure the TAR files are present.")
    # folder to store the intermediate dataset cache
    print("Starting data preprocessing...")
    dataset_path = dotenv_manager.get_dataset_directory()
    temp_dataset_path = os.path.join(dataset_path, "temp", "development")
    # Ensure the intermediate dataset cache exists before writing any shards.
    os.makedirs(temp_dataset_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)

    # Step 1: turn each TAR archive into a saved dataset shard on disk.
    generate_dataset_from_tars(temp_dataset_path, tar_folder, num_proc=num_proc, skip_existing=skip_existing)

    full_ds = []
    # Step 2: reload every shard and keep the partial datasets in memory long
    # enough to concatenate them into a single dataset object.
    for last_child in os.listdir(temp_dataset_path):
        traj_path = os.path.join(temp_dataset_path, last_child)

        print(f"Loading dataset for traj: {last_child}...")
        partial_ds = load_from_disk(traj_path)
        full_ds.append(partial_ds)

    # Merge all trajectory-level shards into one dataset before shuffling and
    # splitting. This gives the training pipeline a single consistent dataset view.
    full_ds = concatenate_datasets(full_ds)
    print(f"✓ Full dataset loaded with {len(full_ds)} samples.")

    # Step 3: shuffle once with a fixed seed so the downstream split is
    # deterministic and not correlated with the ordering of the source TARs.
    full_set = full_ds.shuffle(seed=42)

    # Create the project splits using the shared helper so the split ratios stay
    # aligned with the rest of the training codebase.
    train_val_test = train_val_test_split(full_set, train_fraction=0.7, val_fraction=0.15, seed=42)
    full_training_set = train_val_test['train']
    full_val_set = train_val_test['val']
    full_test_set = train_val_test['test']

    # Log the final split sizes so preprocessing output can be verified quickly.
    print(f"\tFinal training set size: {len(full_training_set)}")
    print(f"\tFinal validation set size: {len(full_val_set)}")
    print(f"\tFinal test set size: {len(full_test_set)}")

    print("Saving final development datasets to disk...")

    # Persist each split separately so training scripts can load only the subset
    # they need without reading the entire corpus every time.
    for split_name, dset in train_val_test.items():
        set_path = os.path.join(dataset_path, split_name)
        os.makedirs(set_path, exist_ok=True)

        if not regenerate and os.path.exists(set_path):
            if "dataset_info.json" in os.listdir(set_path):
                print(f"{split_name.capitalize()} dataset already exists at {set_path}. Skipping regeneration as regenerate is False.")
                continue
        dset.save_to_disk(set_path, max_shard_size="4GB", num_proc=num_proc)

    # remove the temporary dataset cache to free up disk space after the final splits are saved.
    if os.path.exists(temp_dataset_path) and remove_temp_dataset:
        shutil.rmtree(temp_dataset_path)
        print(f"Removed temporary dataset cache at {temp_dataset_path}.")
    print("Datasets saved.")


if __name__ == "__main__":
    # Run the full preprocessing pipeline when this module is executed directly.
    # ``regenerate`` forces shard recreation, while ``skip_existing`` can be used
    # to avoid repeating work on archives that were already processed.

    # Generate the full dataset from TAR files
    # 2 cpus do 1 targz file in ~30 minutes. ~130 targz files in total = ~65 hours.
    # To speed up the process, we can use 32 cpus to process multiple targz files in parallel. = ~4 hours.
    generate_development_dataset_from_tars(regenerate=False, num_proc=16, skip_existing=True)

