from src.models.SCNN import SCNN
from src.models.OneLayer import OneLayerNet
from src.models.DCNN import DCNN
from src.models.LongSequenceAtomTransformer import LongSequenceAtomTransformer
from src.data.data_loading.HFDataloader import initialize_dataloaders
from src.training.batch_preprocessing import prepare_model_batch, prepare_sequence_batch
from src.config.training_config import TrainingConfig
import os 
import torch
import json
import re 
import ast


MODEL_REGISTRY = {
	"SCNN": SCNN,
	"OneLayerNet": OneLayerNet,
	"DCNN": DCNN,
	"LongSequenceAtomTransformer": LongSequenceAtomTransformer,
}

BATCH_PREPARATION_REGISTRY = {
	"LongSequenceAtomTransformer": prepare_sequence_batch,
}



def _build_test_dataloader(config: TrainingConfig, splits=["test"]):
	dataloaders = initialize_dataloaders(config, keep_all_columns=True, splits=splits)
	return dataloaders


def _resolve_batch_preparation_fn(config: TrainingConfig, model_name: str):
	if callable(getattr(config, "batch_preparation_fn", None)):
		return config.batch_preparation_fn
	return BATCH_PREPARATION_REGISTRY.get(model_name, prepare_model_batch)


def _extract_class_name(class_value) -> str | None:
	if class_value is None:
		return None
	if isinstance(class_value, type):
		return class_value.__name__
	if isinstance(class_value, str):
		match = re.search(r"<class '.*\.([^\.']+)'>", class_value)
		if match:
			return match.group(1)
		return class_value.split(".")[-1].strip()
	return None


def _load_config_from_artifacts(config_path: str | None, checkpoint_path: str) -> TrainingConfig | None:
	checkpoint_dir = os.path.dirname(checkpoint_path)
	training_config_path = checkpoint_dir + "/training_config.json"

	if os.path.exists(training_config_path):
		return TrainingConfig.load(training_config_path)
	
	return None


def _resolve_model_class(model_class_name: str):
	if model_class_name in MODEL_REGISTRY:
		return MODEL_REGISTRY[model_class_name]
	supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
	raise ValueError(f"Unknown model class '{model_class_name}'. Supported classes: {supported}")


def _load_state_dict(checkpoint_path: str, device: torch.device) -> dict:
	raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
	if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
		raw_checkpoint = raw_checkpoint["state_dict"]
	if not isinstance(raw_checkpoint, dict):
		raise ValueError("Unsupported checkpoint format. Expected state_dict or dict containing state_dict.")
	return raw_checkpoint


def _load_weights(model: torch.nn.Module, state_dict: dict) -> None:
	try:
		model.load_state_dict(state_dict)
		return
	except RuntimeError:
		pass

	compiled_prefix = "_orig_mod."
	stripped = {
		(key[len(compiled_prefix):] if key.startswith(compiled_prefix) else key): value
		for key, value in state_dict.items()
	}
	model.load_state_dict(stripped)

