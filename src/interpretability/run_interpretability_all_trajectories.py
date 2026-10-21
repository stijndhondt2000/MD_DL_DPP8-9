import multiprocessing as mp
import os
import time
from pathlib import Path

import contextlib
from src.interpretability.aggregate_replicas import count_significant_residues
from src.interpretability.interpretability_utils import CaptumInterpreter, _initilize_classification_model, solve_methods
import torch

from src.interpretability.interpretability_executer import execute_interpretability, save_attribution_colored_pdbs_spatial_scores, write_coloring_script
from src.utils.labels import ligant_to_class
from dotenv import load_dotenv 
import numpy as np
import matplotlib.pyplot as plt
# TODO : add a function to check if the interpretability results for a given output directory already exist, and if so, skip the job


THRESHOLD = 0.5

METHOD_ARGS = {
    # "integrated_gradients": {"steps": 50},
    # "occlusion": {
    #     "patch_size": 1,
    #     "perturbations_per_eval": 2048,
    #     "shift_size": 1,
    # },
    "saliency": {},
}

MODEL_CHECKPOINTS = {
    # "DCNN": Path(
    #     "/project_antwerp/MD_DL_DPP8-9/"
    #     "output/models/CustomDenseNet_Randomsplit_Dataset/"
    #     "20260323-202241/"
    #     "CustomDenseNet_Randomsplit_Dataset.pth"
    # ),
    "SCNN": Path(
        "/project_antwerp/MD_DL_DPP8-9/"
        "output/models/SCNN/"
        "20260815-030746/"
        "SCNN.pth"
    ),
}

def already_completed(output_dir: Path) -> bool:
    """
    Check if the interpretability results for a given output directory already exist.

    :param output_dir: Directory where interpretability results are expected to be saved.
    :return: True if the results already exist, False otherwise.
    """

    expected_files = 208
    # already completed if the images subfolder contains 148 files
    images_dir = output_dir / "images"
    if images_dir.exists() and len(list(images_dir.glob("*"))) >= expected_files:
        return True
    
    return False

def worker(gpu_id, job_queue):
    scnn = None
    dcnn = None
    scnn_methods = None
    dcnn_methods = None

    # Each worker uses its assigned physical GPU.
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")


    if "DCNN" in MODEL_CHECKPOINTS:
        dcnn = _initilize_classification_model(MODEL_CHECKPOINTS["DCNN"], device=device)
        dcnn_methods = solve_methods(CaptumInterpreter(dcnn), method_args=METHOD_ARGS)
    if "SCNN" in MODEL_CHECKPOINTS:
        scnn = _initilize_classification_model(MODEL_CHECKPOINTS["SCNN"], device=device)
        scnn_methods = solve_methods(CaptumInterpreter(scnn), method_args=METHOD_ARGS)


    model=None
    methods=None
    while True:
        job = job_queue.get()

        if job is None:
            break
        (
            replica_dir,
            output_dir,
            binding_type,
            model_name,
            checkpoint,
        ) = job

        start_time = time.time()

        print(
            f"[GPU {gpu_id}] START "
            f"{model_name} | {replica_dir}",
            flush=True,
        )
        
        try:
            if model_name == "DCNN":
                model, methods = dcnn, dcnn_methods
            elif model_name == "SCNN":
                model, methods = scnn, scnn_methods
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            if model is None or methods is None:
                raise RuntimeError(
                    f"{model_name} or its interpretability methods were not initialized"
                )
            # disable printing from the interpretability function to avoid cluttering the output
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                execute_interpretability(
                    pdb_directory=replica_dir,
                    output_dir=output_dir,
                    binding_type=binding_type,
                    model_checkpoint=model,
                    methods=methods,
                    threshold=THRESHOLD,
                )

            elapsed = time.time() - start_time

            print(
                f"[GPU {gpu_id}] DONE "
                f"{model_name} | {replica_dir} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

        except Exception as e:
            print(
                f"[GPU {gpu_id}] ERROR "
                f"{model_name} | {replica_dir}: {e}",
                flush=True,
            )



def execute_model_interpretation(output_root_dir: Path = None, input_dir: Path = None):
    """
    Execute the model interpretation for all trajectories.

    :param output_root_dir: The directory where the output will be saved.
    :param input_dir: The directory where the input data is located.
    """
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    print(f"Using {num_gpus} GPUs")

    ctx = mp.get_context("spawn")

    job_queue = ctx.Queue()

    # Create jobs: (replica, model)
    jobs = []

    for dpp_dir in input_dir.iterdir():
        if not dpp_dir.is_dir():
            continue

        dpp = dpp_dir.stem
        if dpp not in ["DPP8", "DPP9"]:
            print(f"Skipping {dpp_dir} (not DPP8 or DPP9)")
            continue
        for ligand_dir in dpp_dir.iterdir():
            if not ligand_dir.is_dir():
                continue

            ligand = ligand_dir.stem
            binding_type = ligant_to_class(ligand_name=ligand)

            for replica_dir in ligand_dir.iterdir():
                if not replica_dir.is_dir():
                    continue

                replica = replica_dir.stem

                for model_name, checkpoint in MODEL_CHECKPOINTS.items():

                    output_dir = (
                        output_root_dir
                        / dpp
                        / ligand
                        / replica
                        / model_name
                        / "images"
                    )
                    if already_completed(output_dir):
                        print(
                            f"Skipping {model_name} | {replica_dir} "
                            f"(already completed)",
                            flush=True,
                        )
                        continue
                    jobs.append(
                        (
                            replica_dir,
                            output_dir,
                            binding_type,
                            model_name,
                            checkpoint,
                        )
                    )

    print(f"Total jobs: {len(jobs)}")

    # Put all jobs into the shared queue.
    for job in jobs:
        job_queue.put(job)

    # One worker per GPU.
    workers = []

    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=worker,
            args=(gpu_id, job_queue),
        )
        p.start()
        workers.append(p)

    # One sentinel per worker.
    for _ in range(num_gpus):
        job_queue.put(None)

    # Wait for workers.
    for p in workers:
        p.join()



if __name__ == "__main__":
    # read OUTPUT_DIR from .env file 
    load_dotenv()
    root_input_dir = Path("/project_antwerp/dataset/decompressed_both_datasets_every1000/")
    root_output_dir = Path(os.getenv("OUTPUT_DIR", default="/project_antwerp/MD_DL_DPP8-9/output/"))
    interpretability_output_dir = root_output_dir / "interpretability"
    all_frames_output_dir = interpretability_output_dir / "all_frames"
    aggregated_replicas_output_dir = interpretability_output_dir / "aggregated_replicas"
    os.makedirs(all_frames_output_dir, exist_ok=True)
    os.makedirs(aggregated_replicas_output_dir, exist_ok=True)
    execute_model_interpretation(
        input_dir=root_input_dir,
        output_root_dir=all_frames_output_dir

    )
    count_significant_residues(
        input_folder=all_frames_output_dir,
        output_folder_root=aggregated_replicas_output_dir,
        base_pdb_file_folder=root_input_dir,
        methods=list(METHOD_ARGS.keys())
    )