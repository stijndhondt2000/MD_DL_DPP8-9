"""Deterministic replica holdouts for correlated molecular-dynamics frames."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


SPLIT_STRATEGY = "trajectory_holdout_v1"
SPLIT_NAMES = ("train", "val", "test")
GROUP_COLUMNS = ("dpp_class", "ligand_name", "binding_type", "replica_id")
MANIFEST_FILENAME = "split_manifest.json"


def split_trajectories(dataset, train_fraction=0.7, val_fraction=0.15, seed=42):
    """Return frame datasets and an audit manifest, holding out whole replicas.

    Replicas are allocated separately within each (protein, ligand, binding
    type) condition. Every condition needs at least three replicas so it can
    appear in every partition. Fractions target trajectory counts, not frames;
    each partition receives at least one replica per condition. Rows within
    each partition are shuffled only after membership has been fixed, avoiding
    trajectory-ordered prefixes when training/evaluation use a dataset fraction.

    Only metadata columns are read while grouping; coordinates remain backed
    by the original Hugging Face dataset until the selected splits are saved.
    """
    fractions = (train_fraction, val_fraction, 1 - train_fraction - val_fraction)
    if any(not math.isfinite(fraction) or fraction <= 0 for fraction in fractions):
        raise ValueError("Train, validation and remaining test fractions must be positive and finite.")
    missing = set(GROUP_COLUMNS) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Trajectory splitting requires columns: {sorted(missing)}")
    if not len(dataset):
        raise ValueError("Cannot split an empty trajectory dataset.")

    frame_indices = defaultdict(list)
    offset = 0
    for batch in dataset.select_columns(list(GROUP_COLUMNS)).iter(batch_size=10000):
        for values in zip(*(batch[column] for column in GROUP_COLUMNS)):
            if values[0] in (None, "") or values[3] in (None, ""):
                raise ValueError(f"Missing protein or replica identity at row {offset}.")
            # Normalize numeric replica IDs without conflating unrelated conditions.
            key = tuple(None if value is None else str(value) for value in values)
            frame_indices[key].append(offset)
            offset += 1

    conditions = defaultdict(list)
    for key in frame_indices:
        conditions[key[:3]].append(key)
    insufficient = {key: len(groups) for key, groups in conditions.items() if len(groups) < 3}
    if insufficient:
        raise ValueError(
            "Whole-trajectory train/val/test splitting needs at least 3 replicas "
            f"per molecular condition; found {insufficient}. Add replicas or explicitly "
            "redesign the evaluation; random frame splitting is not a fallback."
        )

    rng = random.Random(seed)
    assignments = {name: [] for name in SPLIT_NAMES}
    for condition in sorted(conditions, key=json.dumps):
        groups = sorted(conditions[condition], key=json.dumps)
        rng.shuffle(groups)
        counts = [1, 1, 1]
        targets = [len(groups) * fraction for fraction in fractions]
        # Allocate remaining replicas to the most underfilled target. Ties use
        # train/val/test order, so small-group rounding is reproducible.
        for _ in range(len(groups) - 3):
            index = max(range(3), key=lambda i: round(targets[i] - counts[i], 12))
            counts[index] += 1
        start = 0
        for name, count in zip(SPLIT_NAMES, counts):
            assignments[name].extend(groups[start:start + count])
            start += count

    assigned = [key for groups in assignments.values() for key in groups]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(frame_indices):
        raise RuntimeError("Trajectory assignments overlap or omit trajectories.")
    expected_conditions = set(conditions)
    if any({key[:3] for key in groups} != expected_conditions for groups in assignments.values()):
        raise RuntimeError("A partition is missing molecular conditions.")

    splits = {}
    manifest = {
        "strategy": SPLIT_STRATEGY,
        "seed": seed,
        "target_trajectory_fractions": dict(zip(SPLIT_NAMES, fractions)),
        "group_columns": list(GROUP_COLUMNS),
        "condition_columns": list(GROUP_COLUMNS[:3]),
        "source_fingerprint": dataset._fingerprint,
        "shuffle_within_splits": True,
        "splits": {},
    }
    for name, groups in assignments.items():
        indices = sorted(index for key in groups for index in frame_indices[key])
        splits[name] = dataset.select(indices).shuffle(seed=seed)
        manifest["splits"][name] = {
            "frame_count": len(indices),
            "trajectory_count": len(groups),
            "trajectories": [dict(zip(GROUP_COLUMNS, key)) for key in groups],
        }
    return splits, manifest


def read_split_manifest(dataset_path):
    """Reject legacy or incomplete development outputs before cache conversion."""
    path = Path(dataset_path) / MANIFEST_FILENAME
    if not path.is_file():
        raise ValueError(f"Missing {path}. Regenerate the development splits with --regenerate-splits.")
    manifest = json.loads(path.read_text())
    if manifest.get("strategy") != SPLIT_STRATEGY:
        raise ValueError(f"Unsupported split strategy in {path}; regenerate with --regenerate-splits.")
    if set(manifest.get("splits", {})) != set(SPLIT_NAMES):
        raise ValueError(f"Incomplete split manifest: {path}")
    return manifest


def manifest_id(manifest):
    """Identify one complete split assignment for safetensor provenance."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def cache_provenance(dataset_fingerprint, manifest=None):
    """Bind tensor shards to their input rows and trajectory assignment."""
    return {
        "source_fingerprint": dataset_fingerprint,
        "split_strategy": manifest["strategy"] if manifest else None,
        "split_manifest_id": manifest_id(manifest) if manifest else None,
    }


def validate_development_cache(metadata):
    """Fail closed for old random-frame caches, including direct dataset loads."""
    if metadata.get("split_strategy") != SPLIT_STRATEGY or not metadata.get("split_manifest_id"):
        raise ValueError(
            "This development safetensor cache has no trajectory-holdout provenance. "
            "Run the complete preprocessing pipeline with --regenerate-splits before training."
        )
