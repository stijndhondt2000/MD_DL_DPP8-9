"""Run with ``python -m unittest discover -s tests -v``.

Core invariants run without the GPU/scientific stack. When datasets is installed,
the same behavior is also exercised with a real disk-backed Hugging Face dataset.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

from src.data.data_preprocessing.trajectory_split import (
    GROUP_COLUMNS, MANIFEST_FILENAME, SPLIT_NAMES, cache_provenance,
    manifest_id, read_split_manifest, split_trajectories, validate_development_cache,
)


class MemoryDataset:
    """Small dataset adapter for tests without optional scientific dependencies."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else list(GROUP_COLUMNS)
        self._fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def select_columns(self, columns):
        return MemoryDataset([{key: row[key] for key in columns} for row in self.rows])

    def iter(self, batch_size):
        # Grouping must never materialize the expensive coordinate arrays.
        if set(self.column_names) != set(GROUP_COLUMNS):
            raise AssertionError("Grouping read non-metadata columns")
        for start in range(0, len(self), batch_size):
            rows = self.rows[start:start + batch_size]
            yield {key: [row[key] for row in rows] for key in self.column_names}

    def select(self, indices):
        return MemoryDataset([self.rows[index] for index in indices])

    def shuffle(self, seed):
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return MemoryDataset(rows)


def sample_rows(replicas=3):
    rows = []
    for protein in ("dpp8", "dpp9"):
        for ligand, binding_type in (("12i", "nonbinder"), ("42", "dpp9selective")):
            for replica in range(replicas):
                # Unequal lengths ensure assertions operate on trajectories,
                # rather than accidentally relying on equal frame counts.
                for frame in range(2 + replica):
                    rows.append({
                        "dpp_class": protein, "ligand_name": ligand,
                        "binding_type": binding_type, "replica_id": str(replica),
                        "pdb_id": f"{protein}/{ligand}/{replica}/{frame}",
                        "coordinates": [[float(frame), 0.0, 0.0]],
                    })
    return rows


def trajectory_keys(rows):
    return {tuple(row[column] for column in GROUP_COLUMNS) for row in rows}


class TrajectorySplitTests(unittest.TestCase):
    def test_whole_trajectories_no_frame_loss_and_condition_coverage(self):
        rows = sample_rows(5)
        splits, manifest = split_trajectories(MemoryDataset(rows))
        expected_conditions = {key[:3] for key in trajectory_keys(rows)}
        seen_frames, seen_groups = set(), set()
        for name in SPLIT_NAMES:
            result = splits[name].rows
            groups = trajectory_keys(result)
            frames = {row["pdb_id"] for row in result}
            self.assertTrue(groups.isdisjoint(seen_groups))
            self.assertTrue(frames.isdisjoint(seen_frames))
            self.assertEqual({key[:3] for key in groups}, expected_conditions)
            self.assertEqual(frames, {row["pdb_id"] for row in rows if tuple(row[c] for c in GROUP_COLUMNS) in groups})
            self.assertEqual(manifest["splits"][name]["frame_count"], len(result))
            self.assertEqual(manifest["splits"][name]["trajectory_count"], len(groups))
            seen_groups.update(groups)
            seen_frames.update(frames)
        self.assertEqual(seen_frames, {row["pdb_id"] for row in rows})

    def test_three_replicas_give_one_per_condition_per_split(self):
        splits, _ = split_trajectories(MemoryDataset(sample_rows(3)))
        for split in splits.values():
            self.assertEqual(len(trajectory_keys(split.rows)), 4)

    def test_ten_replicas_round_to_seven_two_one(self):
        _, manifest = split_trajectories(MemoryDataset(sample_rows(10)))
        self.assertEqual([manifest["splits"][name]["trajectory_count"] for name in SPLIT_NAMES], [28, 8, 4])

    def test_seed_is_reproducible_and_membership_ignores_input_order(self):
        dataset = MemoryDataset(sample_rows(8))
        first, manifest = split_trajectories(dataset, seed=21)
        repeated, repeated_manifest = split_trajectories(dataset, seed=21)
        reordered, _ = split_trajectories(dataset.shuffle(seed=999), seed=21)
        self.assertEqual(manifest, repeated_manifest)
        for name in SPLIT_NAMES:
            self.assertEqual(first[name].rows, repeated[name].rows)
            self.assertEqual(trajectory_keys(first[name].rows), trajectory_keys(reordered[name].rows))
        changed, _ = split_trajectories(dataset, seed=22)
        self.assertNotEqual(trajectory_keys(first["test"].rows), trajectory_keys(changed["test"].rows))

    def test_each_partition_is_shuffled_after_assignment(self):
        rows = sample_rows(5)
        splits, _ = split_trajectories(MemoryDataset(rows))
        for name in SPLIT_NAMES:
            ids = {row["pdb_id"] for row in splits[name].rows}
            self.assertNotEqual(splits[name].rows, [row for row in rows if row["pdb_id"] in ids])

    def test_insufficient_replicas_fail_instead_of_splitting_frames(self):
        with self.assertRaisesRegex(ValueError, "at least 3 replicas"):
            split_trajectories(MemoryDataset(sample_rows(2)))

    def test_invalid_fractions_fail(self):
        for train, val in ((0, .2), (.7, 0), (.8, .3), (-.1, .2), (float("nan"), .2), (.7, float("inf"))):
            with self.subTest(train=train, val=val), self.assertRaises(ValueError):
                split_trajectories(MemoryDataset(sample_rows()), train, val)

    def test_missing_metadata_and_empty_data_fail(self):
        rows = sample_rows()
        del rows[0]["replica_id"]
        with self.assertRaisesRegex(ValueError, "requires columns"):
            split_trajectories(MemoryDataset(rows))
        rows = sample_rows()
        rows[0]["replica_id"] = None
        with self.assertRaisesRegex(ValueError, "Missing protein or replica"):
            split_trajectories(MemoryDataset(rows))
        with self.assertRaisesRegex(ValueError, "empty"):
            split_trajectories(MemoryDataset([]))

    def test_replica_ids_can_repeat_across_conditions(self):
        rows = sample_rows()
        for row in rows:
            row["replica_id"] = int(row["replica_id"])
        splits, _ = split_trajectories(MemoryDataset(rows))
        self.assertEqual(sum(len(split) for split in splits.values()), len(rows))

    def test_grouping_across_metadata_batch_boundaries(self):
        rows = sample_rows()
        rows = [dict(row, pdb_id=f"{i}/{row['pdb_id']}") for i in range(300) for row in rows]
        splits, _ = split_trajectories(MemoryDataset(rows))
        groups = [trajectory_keys(splits[name].rows) for name in SPLIT_NAMES]
        self.assertTrue(groups[0].isdisjoint(groups[1] | groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_manifest_roundtrip_and_legacy_rejection(self):
        _, manifest = split_trajectories(MemoryDataset(sample_rows()))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "regenerate"):
                read_split_manifest(directory)
            path = Path(directory) / MANIFEST_FILENAME
            path.write_text(json.dumps(manifest))
            self.assertEqual(read_split_manifest(directory), manifest)
            manifest["strategy"] = "random_frames"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_split_manifest(directory)

    def test_provenance_changes_when_rows_or_assignments_change(self):
        _, manifest = split_trajectories(MemoryDataset(sample_rows()), seed=42)
        _, changed = split_trajectories(MemoryDataset(sample_rows()), seed=43)
        provenance = cache_provenance("frames-a", manifest)
        validate_development_cache(provenance)
        self.assertNotEqual(provenance, cache_provenance("frames-b", manifest))
        self.assertNotEqual(provenance, cache_provenance("frames-a", changed))
        self.assertEqual(manifest_id(manifest), manifest_id(json.loads(json.dumps(manifest))))
        with self.assertRaisesRegex(ValueError, "no trajectory-holdout provenance"):
            validate_development_cache({"total_samples": 100, "split": "train"})

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "Hugging Face datasets is not installed")
    def test_real_huggingface_save_reload(self):
        from datasets import Dataset, load_from_disk

        rows = sample_rows(5)
        splits, manifest = split_trajectories(Dataset.from_list(rows))
        seen = set()
        with tempfile.TemporaryDirectory() as directory:
            for name, dataset in splits.items():
                dataset.save_to_disk(str(Path(directory) / name))
                loaded = load_from_disk(str(Path(directory) / name))
                groups = trajectory_keys(loaded)
                self.assertTrue(groups.isdisjoint(seen))
                seen.update(groups)
                self.assertEqual(len(loaded), manifest["splits"][name]["frame_count"])
        self.assertEqual(seen, trajectory_keys(rows))


if __name__ == "__main__":
    unittest.main()
