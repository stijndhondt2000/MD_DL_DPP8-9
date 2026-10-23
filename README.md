# Thesis: Molecular Dynamics Trajectory Embeddings

Short README for the repository used in the Master's thesis project on learning embeddings for molecular dynamics trajectories.

**Contents:**
- **Description:** high-level project goals and approach
- **Getting started:** environment and dependencies
- **Usage:** data, training, and experiments
- **Project structure:** key folders and files
- **Notes:** tips for common tasks

**Description**:
- **Goal:** Build and evaluate trajectory-level embeddings for molecular dynamics data to support downstream tasks (classification, clustering, interpretability).
- **Approach:** Preprocess molecular trajectory data, train models (CNNs and other architectures) to produce embeddings, and analyze downstream performance and interpretability.

**Getting started**:
- Python: recommended 3.11/3.12 (project contains a `gpulab_env` virtualenv configured for Python 3.12).
- Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- If you prefer the interpretability environment, use `requirements-interpretability.txt`.

**API documentation**:
- Install the documentation dependency with `pip install -r requirements-docs.txt`.
- Generate HTML documentation from the `src/` docstrings with:

```bash
pdoc --output-directory docs/api src
```

- Open `docs/api/src/index.html` after generation.

**Running experiments**:
- A convenience script `run_experiment.sh` exists for running typical experiments. Many experiments are implemented under `src/experiments/`.
- Example: run the simple CNN experiment (uses the workspace venv or Docker task):

```bash
python src/experiments/SCNN.py
```

**Data**:
- Cached and preprocessed datasets live under `hf_cache/` and mapped Arrow files under `hf_cache/mapped_cache/`.
- Preprocessing code is in `src/data/data_preprocessing/` and `src/data/data_loading/`.

**Trajectory-aware evaluation**:
- Development data now uses whole-trajectory holdouts, replacing random frame splitting.
  A trajectory is identified by `(dpp_class, ligand_name, binding_type, replica_id)`.
  Every trajectory belongs to exactly one of `train`, `val`, or `test`.
- Replicas are assigned separately within each `(dpp_class, ligand_name, binding_type)`
  condition, with seed 42. At least three replicas per condition are required;
  insufficient replicas raise an error rather than falling back to frame splitting.
- The target 70%/15%/15% applies to trajectory counts, with at least one trajectory
  per condition in each partition. Three replicas give 1/1/1; ten give 7/2/1.
  Unequal trajectory lengths mean frame proportions can differ further.
- Rows are shuffled **after** membership is fixed, separately within each partition.
  This avoids trajectory-ordered prefixes for experiments using `dataset_size < 1`;
  small sampled prefixes are still not guaranteed to contain every class.
- `DATASET_DIR/split_manifest.json` records the seed, strategy, source fingerprint,
  complete trajectory assignments, and frame/trajectory counts. Tensor caches record
  the manifest ID and input fingerprint. Changed or legacy caches are rewritten;
  training rejects legacy caches and mixed split assignments. The training name
  `validation` maps to the saved `val` partition.
- This evaluates unseen replicas of represented molecular conditions. It is not a
  chronological forecast or an unseen-ligand evaluation, and it does not establish
  independence of replicas that share initial structures or simulation ancestry.

To migrate existing random-frame outputs, use the configured data environment
(`.env` with `DATASET_DIR`, `SAFETENSOR_DIR`, `COMPRESSED_DATASET_DIR`, and
`COMPRESSED_EXTERNAL_DATASET_DIR`) and run from the repository root:

```bash
python -m src.data.data_preprocessing.complete_data_preprocessing_pipeline --regenerate-splits
```

This replaces all three development splits and rebuilds their tensor caches.
Existing per-trajectory intermediates can be reused. The external dataset remains
separate. Without the flag, legacy/incompatible development outputs are rejected.
Retrain models after migration; old random-frame metrics are not directly comparable.

Regression checks:

```bash
python -m unittest discover -s tests -v
```

Core split checks use a small in-memory dataset adapter; cache and loader checks
mock tensor IO/dependencies. The real Hugging Face save/reload integration check
runs when `datasets` is installed.

**Project structure (high level)**:
- `src/` : source code
  - `data_preprocessing/` : preprocessing pipelines (e.g., `pre_processing_huggingface.py`)
  - `data_loading/` : dataset loaders
  - `experiments/` : experiment entry points and scripts
  - `model_training/`, `Models/` : model implementations and training utilities
  - `utils/` : helper utilities and common functions
  - `interpretability/`: interpretability scripts
  - `models/`: model definitions and architectures 
- `hf_cache/` : local cache of datasets / transformed Arrow shards
- `jupyter_notebooks/` : exploratory analysis and debugging notebooks
- `output/` : trained models, plots, and profiling outputs
- `CLI/` : scripts for running command-line experiments 
