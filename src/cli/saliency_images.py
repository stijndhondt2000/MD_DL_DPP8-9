"""Run three attribution methods on two saved training images by default.

Run from the repository root with python -m src.cli.saliency_images --help.
Inputs are image safetensors produced by preprocess_images, not coordinates.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from matplotlib import colormaps
from safetensors import safe_open

from src.interpretability.sanity_checks import load_model
from src.interpretability.interpretability_utils import CaptumInterpreter


def load_images(directory, limit):
    """Read only the requested number of training samples, in sorted file order."""
    samples = []
    for path in sorted(directory.rglob('*.safetensors')):
        with safe_open(str(path), framework='pt', device='cpu') as f:
            metadata = f.metadata() or {}
            source = metadata.get('source_shard', str(path))
            if not any(p == 'train' or p.startswith('train_shard_') for p in Path(source).parts):
                continue
            if not {'images', 'labels', 'num_atoms'}.issubset(f.keys()):
                raise ValueError(f'{path}: expected images, labels, num_atoms; use saved image files')
            shape = f.get_slice('images').get_shape()
            if len(shape) != 4 or shape[1:] != [3, 168, 168]:
                raise ValueError(f'{path}: expected [N, 3, 168, 168], got {shape}')
            for key in ('labels', 'num_atoms'):
                if f.get_slice(key).get_shape() != [shape[0]]:
                    raise ValueError(f'{path}: invalid {key} shape')
            take = min(limit - len(samples), shape[0])
            images = f.get_slice('images')[:take].clone()
            labels = f.get_slice('labels')[:take]
            counts = f.get_slice('num_atoms')[:take]
            for i in range(take):
                x = images[i:i+1].contiguous()
                n = int(counts[i])
                if x.dtype != torch.float32 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                    raise ValueError(f'{path}: expected finite float32 images in [0, 1]')
                if not 0 < n <= 168 * 168 or torch.count_nonzero(x.flatten(2)[:, :, n:]):
                    raise ValueError(f'{path}: invalid atom count or padding')
                samples.append(dict(images=x, label=int(labels[i]), num_atoms=n,
                                    tensor_file=str(path.resolve()), index_in_file=i,
                                    source_shard=source,
                                    source_index=int(metadata.get('start', '0')) + i))
        if len(samples) == limit:
            return samples
    raise ValueError(f'Found {len(samples)} training images in {directory}; need {limit}')


def run_methods(model, x, target, steps, perturbations):
    """Explain the same predicted-class logit with all three methods."""
    interpreter = CaptumInterpreter(model)
    results = {}
    print('  Integrated Gradients', flush=True)
    ig = interpreter.integrated_gradients(
        x.detach().clone().requires_grad_(True), target=target,
        baselines=torch.zeros_like(x), n_steps=steps, internal_batch_size=1,
    )
    results['integrated_gradients'] = ig.attributions.detach().cpu()
    results['ig_convergence_delta'] = ig.convergence_delta.detach().cpu()

    print('  Saliency', flush=True)
    saliency = interpreter.saliency(
        x.detach().clone().requires_grad_(True), target=target, abs_values=False,
    )
    results['saliency'] = saliency.attributions.detach().cpu()

    print('  Occlusion (one coordinate channel at a time; this can be slow)', flush=True)
    with torch.no_grad():
        occlusion = interpreter.occlusion(
            x.detach().clone(), target=target, patch_size=1, shift_size=1,
            perturbations_per_eval=perturbations,
        )
    results['occlusion'] = occlusion.attributions.detach().cpu()
    for name in ('integrated_gradients', 'saliency', 'occlusion'):
        if results[name].shape != x.shape or not torch.isfinite(results[name]).all():
            raise ValueError(f'{name}: invalid attribution output')
    return results


def save_pngs(sample, results, directory):
    """Save previews on the machine running the job; keep raw values in .pt."""
    directory.mkdir(parents=True, exist_ok=False)
    pixels = (sample['images'][0].detach().cpu().permute(1, 2, 0)
              .clamp(0, 1).mul(255).round().to(torch.uint8).numpy())
    Image.fromarray(pixels).save(directory / 'input.png')
    scales = {}
    for method in ('integrated_gradients', 'saliency', 'occlusion'):
        # Channel magnitude per atom. Exclude padding from display and scaling.
        scores = results[method][0].detach().cpu().abs().sum(0).clone()
        scores.flatten()[sample['num_atoms']:] = 0
        maximum = float(scores.max())
        normalized = scores / maximum if maximum > 0 else torch.zeros_like(scores)
        colors = colormaps['inferno'](normalized.numpy(), bytes=True)[..., :3].copy()
        colors.reshape(-1, 3)[sample['num_atoms']:] = 0
        Image.fromarray(colors).save(directory / f'{method}.png')
        scales[method] = maximum
    (directory / 'display_settings.json').write_text(json.dumps(dict(
        score='sum of absolute channel attributions', normalization='per-map maximum',
        maxima=scales, padding='black', colormap='inferno',
    ), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trusted model checkpoint')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-samples', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--perturbations', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if min(args.max_samples, args.steps, args.perturbations) < 1:
        parser.error('Sample count, steps, and perturbations must be positive')
    if args.output_dir.exists():
        parser.error('Use a new output directory')

    torch.manual_seed(42)
    samples = load_images(args.image_dir, args.max_samples)
    model = load_model(args.checkpoint, args.device)  # Sets eval mode.
    args.output_dir.mkdir(parents=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(seed=42, target='original predicted class', baseline='zero',
                    occlusion_window=[1, 1, 1], split='train')
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2))

    for i, sample in enumerate(samples):
        x = sample['images'].to(args.device)
        with torch.no_grad():
            logits = model(x)
        target = int(logits.argmax(1).item())
        print(f'Sample {i+1}/{len(samples)}: label={sample["label"]}, target={target}', flush=True)
        results = run_methods(model, x, target, args.steps, args.perturbations)
        target_file = args.output_dir / f'sample_{i:03d}.pt'
        temporary = target_file.with_suffix('.part')
        torch.save(dict(sample, target=target, logits=logits.cpu(), results=results), temporary)
        temporary.replace(target_file)
        png_dir = args.output_dir / 'png' / f'sample_{i:03d}'
        save_pngs(sample, results, png_dir)
        print(f'Saved {target_file}', flush=True)
        print(f'Saved PNG previews to {png_dir}', flush=True)


if __name__ == '__main__':
    main()
