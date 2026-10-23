"""Exercise real cache orchestration with tensor IO/dependencies mocked."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import tempfile

from src.data.data_preprocessing.trajectory_split import SPLIT_STRATEGY


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.torch = MagicMock()
        self.torch.load.side_effect = lambda path: json.loads(Path(path).read_text())
        self.torch.save.side_effect = lambda value, path: Path(path).write_text(json.dumps(value))
        mocks = {
            "torch": self.torch,
            "dotenv": MagicMock(),
            "src.infrastructure.dotenvManager": MagicMock(),
            "src.training.batch_preprocessing": SimpleNamespace(TARGET_PIXELS=4),
            "src.config.training_config": MagicMock(),
            "src.data.data_loading.SafetensorsDataset": MagicMock(),
            "pyarrow": MagicMock(), "pyarrow.ipc": MagicMock(),
            "datasets": MagicMock(), "numpy": MagicMock(),
            "safetensors": MagicMock(), "safetensors.torch": MagicMock(),
            "huggingface_hub": MagicMock(),
        }
        with patch.dict("sys.modules", mocks), patch("signal.signal"):
            self.module = load_module(
                "src/data/data_preprocessing/prepare_safetensors.py", "cache_under_test"
            )
        self.module.calculate_sample_size = lambda sample: 4
        self.module.safe_open = MagicMock()
        self.dataset = MagicMock()
        self.dataset.__len__.return_value = 4
        self.dataset._fingerprint = "new-frame-membership"
        self.manifest = {"strategy": SPLIT_STRATEGY, "assignment": "new"}

        def write_shard(shard_idx, n_shards, start, end, shard_path, **kwargs):
            Path(shard_path).write_text(self.dataset._fingerprint)

        self.module._write_safetensor_shard_from_worker_split = MagicMock(side_effect=write_shard)
        self.writer = self.module._write_safetensor_shard_from_worker_split
        self.metadata_path = self.directory / "train_metadata.pt"
        self.shards = [self.directory / f"train_shard_{i:04d}_of_0002.safetensors" for i in range(2)]

    def save(self, reuse_existing=True):
        return self.module._save_split_as_safetensors_memory_efficient(
            {"train": self.dataset}, "train", self.directory, shard_size=2,
            split_manifest=self.manifest, reuse_existing=reuse_existing,
        )

    def test_readable_legacy_shards_are_rewritten_even_with_same_sizes(self):
        self.metadata_path.write_text(json.dumps({
            "total_samples": 4, "n_shards": 2, "shard_size": 2, "split": "train",
        }))
        for path in self.shards:
            path.write_text("old random frames")
        self.save()
        self.assertEqual(self.writer.call_count, 2)
        self.assertTrue(all(path.read_text() == self.dataset._fingerprint for path in self.shards))
        metadata = json.loads(self.metadata_path.read_text())
        self.assertEqual(metadata["split_strategy"], SPLIT_STRATEGY)

    def test_matching_shards_are_reused_but_changed_source_is_rebuilt(self):
        self.save()
        self.writer.reset_mock()
        self.save()
        self.writer.assert_not_called()
        self.dataset._fingerprint = "changed-frame-membership"
        self.save()
        self.assertEqual(self.writer.call_count, 2)

    def test_changed_assignment_and_forced_rebuild_rewrite_shards(self):
        self.save()
        self.writer.reset_mock()
        self.manifest["assignment"] = "different-seed"
        self.save()
        self.assertEqual(self.writer.call_count, 2)
        self.writer.reset_mock()
        self.save(reuse_existing=False)
        self.assertEqual(self.writer.call_count, 2)

    def test_interrupted_rebuild_does_not_advertise_completed_metadata(self):
        self.save()
        self.dataset._fingerprint = "different-source"
        self.writer.side_effect = RuntimeError("interrupted write")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.save()
        self.assertFalse(self.metadata_path.exists())


class LoaderSplitTests(unittest.TestCase):
    def make_loader_module(self, ids):
        torch = MagicMock()
        torch.utils.data.DataLoader = lambda dataset, **kwargs: SimpleNamespace(dataset=dataset)
        dataset_factory = MagicMock(side_effect=lambda path, split, **kwargs: SimpleNamespace(
            split=split, split_manifest_id=ids[split],
        ))
        mocks = {
            "torch": torch, "multiprocessing": MagicMock(),
            "src.data.data_loading.SafetensorsDataset": SimpleNamespace(ShardedSafetensorsDataset=dataset_factory),
            "src.infrastructure.dotenvManager": MagicMock(),
            "src.config.training_config": MagicMock(),
            "src.infrastructure.logger": MagicMock(),
        }
        with patch.dict("sys.modules", mocks):
            module = load_module("src/data/data_loading/HFDataloader.py", "loader_under_test")
        return module, dataset_factory

    def test_validation_alias_reads_val_and_preserves_training_api(self):
        module, factory = self.make_loader_module({name: "same-manifest" for name in ("train", "val", "test")})
        config = SimpleNamespace(num_cpus=2, batch_size=4, dataset_size=1.0)
        loaders = module.initialize_dataloaders(config, dotenv_manager=MagicMock())
        self.assertEqual(set(loaders), {"train", "validation", "test"})
        self.assertEqual(loaders["validation"].dataset.split, "val")
        self.assertEqual([call.args[1] for call in factory.call_args_list], ["train", "val", "test"])

    def test_mixed_assignment_caches_are_rejected(self):
        module, _ = self.make_loader_module({"train": "manifest-a", "val": "manifest-b", "test": "manifest-a"})
        config = SimpleNamespace(num_cpus=2, batch_size=4, dataset_size=1.0)
        with self.assertRaisesRegex(ValueError, "different split assignments"):
            module.initialize_dataloaders(config, dotenv_manager=MagicMock())

    def test_direct_sharded_dataset_load_rejects_legacy_metadata(self):
        torch = MagicMock()
        torch.utils.data.Dataset = object
        torch.load.return_value = {"total_samples": 4, "n_shards": 2}
        tensors = MagicMock()
        with patch.dict("sys.modules", {"torch": torch, "safetensors": tensors}):
            module = load_module("src/data/data_loading/SafetensorsDataset.py", "dataset_under_test")
        with self.assertRaisesRegex(ValueError, "no trajectory-holdout provenance"):
            module.ShardedSafetensorsDataset("unused", "train")
        tensors.safe_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
