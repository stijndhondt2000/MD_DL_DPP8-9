"""Convert coordinate shards into batches of float32 image tensors.

Run from the repository root:
    python -m src.cli.preprocess_images --input-dir /path/to/shards --output-dir /path/to/images

Each output safetensors file contains images [B, 3, 168, 168], labels,
and num_atoms. No random augmentation is applied.
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.training.batch_preprocessing import prepare_model_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    source = args.input_dir.resolve()
    output = args.output_dir.resolve()


    files = sorted(source.rglob("*.safetensors"))
    if not files:
        parser.error(f"No safetensors files found in {source}")
    # Refuse to mix a new conversion with existing output.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    total = 0
    with torch.no_grad():
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                keys = ("data", "labels", "num_atoms")
                if not set(keys).issubset(f.keys()):
                    raise ValueError(f"{path} is missing one of {keys}")
                count = f.get_slice("data").get_shape()[0]
                destination = output / path.relative_to(source).with_suffix("")
                destination.mkdir(parents=True, exist_ok=True)
                for start in range(0, count, args.batch_size):
                    end = min(start + args.batch_size, count)
                    batch = {key: f.get_slice(key)[start:end] for key in keys}
                    images, labels = prepare_model_batch(
                        batch, device="cpu", scramble=False
                    )
                    target = destination / f"batch_{start:07d}_{end:07d}.safetensors"
                    temporary = target.with_suffix(".part")
                    save_file(
                        {"images": images, "labels": labels,
                         "num_atoms": batch["num_atoms"].contiguous()},
                        str(temporary),
                        metadata={"source_shard": str(path.relative_to(source)),
                                  "start": str(start), "end_exclusive": str(end),
                                  "scramble": "false"},
                    )
                    temporary.replace(target)
                    total += len(images)
            print(f"Converted {path.name}: {count} images", flush=True)
    print(f"Saved {total} image tensors to {output}")


if __name__ == "__main__":
    main()
