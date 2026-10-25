# Thesis: Molecular Dynamics Trajectory Embeddings

Short README for the repository used in the Master's thesis project on learning embeddings for molecular dynamics trajectories.

**Contents:**
- **Description:** high-level project goals and approach
- **Getting started:** environment and dependencies
- **Usage:** data, training, experiments, interpretability
- **Project structure:** key folders and files
- **Notes:** tips for common tasks

---

## Description
- **Goal:** Build and evaluate trajectory-level embeddings for molecular dynamics data to support downstream tasks (classification, clustering, interpretability).
- **Approach:** Preprocess molecular trajectory data, train models CNNs to produce embeddings, and analyze downstream performance and interpretability.

---

## Getting Started

### Python Environment
- Recommended: Python 3.11/3.12 (project contains an image configured for Python 3.12).
- Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Installation
- Install standard dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- For interpretability tasks, use `requirements-interpretability.txt`.

---

## Usage
### .env file 
The following environment variables should be set in a `.env` file in the project root directory. This is important for running experiments and scripts. 
```sh
COMPRESSED_DATASET_DIR=/project_antwerp/dataset/compressed_dataset_development/ # the directory containing all the tarred (gz) pdb archives
COMPRESSED_EXTERNAL_DATASET_DIR=/project_antwerp/dataset/compressed_dataset_external/
DATASET_DIR=/project_antwerp/dataset/arrow_dataset/ # directory storing the dataset (arrow) files after converting the pdb files
SAFETENSOR_DIR=/project_antwerp/dataset/safetensors/ # tensor files containing dataset and labels ready for training
OUTPUT_DIR=/project_antwerp/MD_DL_DPP8-9/output/ # directory to which to store output (models and interpretability)
HF_HOME=/project_antwerp/temp/hf_cache/ # change the cache directory for huggingface datasets library, important such that gpulab root does not get cluttered (it will stop your job)
HF_DATASETS_DISABLE_PROGRESS_BARS=1 # disable progress bars as gpulab is not interactive and output is messed up
```


### Data Handling
- The data pipeline is implemented in `src/data_preprocessing/` and `src/data_loading/`. 
- The pipeline converts raw molecular dynamics trajectory data (PDB files) in `${COMPRESSED_DATASET_DIR` and `COMPRESSED_EXTERNAL_DATASET_DIR` into a format suitable located  for model training (Arrow datasets in `DATASET_DIR` and Safetensors in `SAFETENSOR_DIR`).

Use the following command to run the preprocessing pipeline (ensure the `.env` file is set up correctly):
```bash
python3 -m src.interpretability.run_interpretability_all_trajectories 


### API Documentation
- Generate HTML documentation from the `src/` docstrings using Sphinx

**Sphinx Documentation**
The project also supports generating comprehensive documentation using Sphinx. See [`docs/docs_readme.md`](docs/docs_readme.md) for detailed instructions on setting up and building Sphinx documentation.

### Running Experiments
- A convenience script `run_experiment.sh` exists for running typical experiments.
- Many experiments are implemented under `src/experiments/`.
- Example: run the simple CNN experiment (uses the workspace venv or Docker task):

```bash
python3 -m src.experiments.SCNN
```



## Project Structure

### Key Folders and Files
- `docs/` : Sphinx documentation
  - `docs/docs_readme.md` : instructions for generating Sphinx documentation
  - `docs/source/` : Sphinx source files
  - `docs/build/` : generated HTML documentation
  - `Dockerimage/` : Docker image files and instructions
- `src/` : source code
  - `data_preprocessing/` : preprocessing pipelines
  - `data_loading/` : dataset loaders
  - `experiments/` : experiment entry points and scripts
  - `model_training/` : model implementations and training utilities
  - `utils/` : helper utilities and common functions
  - `interpretability/` : interpretability scripts
  - `models/` : model definitions and architectures 
- `output/` : trained models, plots, and profiling outputs
---
## Docker Image

### Building the Docker Image

This project provides a Docker image for reproducible environments with PyTorch GPU support. See [`dockerimage/dockerimage_readme.md`](dockerimage/dockerimage_readme.md) for detailed instructions.

## Notes & Tips

### Environment Setup
- Ensure all necessary source code files are in the `src/` directory and importable.
- The `setup_script.sh` can help automate environment setup.

### Quick Commands Reference
- Run data preprocessing pipeline: `python3 -m src.data.data_preprocessing.complete_data_preprocessing_pipeline`
- Start training: `python3 -m src.experiments.SCNN`
- Test trained model: `python3 -m src.experiments.SCNN_test`
- Run interpretability analysis: `python3 -m src.interpretability.run_interpretability_all_trajectories`



### Environment Setup
- Ensure all necessary source code files are in the `src/` directory and importable.
- The `setup_script.sh` can help automate environment setup.

- Example usage: `cat example_usage.txt`
