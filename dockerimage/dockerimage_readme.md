# Updating the DPP8-9 Docker Image

This guide describes how to build a new version of the Docker image for the MD_DL_DPP8-9 project, publish it to GitHub Container Registry (GHCR), and update the `latest` tag.

## 1. Image details

- **Registry:** `ghcr.io`
- **Image:** `ghcr.io/stijndhondt2000/dpp8-9-image`
- **Base image:** `pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime`
- **Project working directory:** `/project_antwerp`

Use versioned tags (e.g. `v1`, `v2`, `v3`) so previous versions remain available.

## 2. Update the Dockerfile

The Dockerfile is located at `dockerimage/DockerFile`.

Notes:
- `WORKDIR` must come before `COPY` so `requirements.txt` is placed in `/project_antwerp`.
- Build from the project root so Docker can access `requirements.txt`.
- PyTorch is installed from the CUDA 13.0 wheel index to support compatible newer NVIDIA GPUs, including Blackwell GPUs. Confirm compatibility with the base image's Python version before upgrading.
- Keep any additional project-specific dependencies in `requirements.txt` or the Dockerfile.

## 3. Build a new version

Run these commands from the project root:

```bash
cd ~/Documents/Projects/MD_DL_DPP8-9

sudo docker build --no-cache \
  -f ./dockerimage/dockerfile \
  -t ghcr.io/stijndhondt2000/dpp8-9-image:v{version_number} .
```

Replace `version_number` with the next version number when creating a later release.

The final `.` specifies the project root as the Docker build context.

## 4. Verify the image

Check the installed PyTorch version:

```bash
sudo docker run --rm \
  --entrypoint python3 \
  ghcr.io/stijndhondt2000/dpp8-9-image:v{version_number} \
  -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

If an NVIDIA GPU is available on the build/test host and NVIDIA Container Toolkit is configured, test GPU access:

```bash
sudo docker run --rm --gpus all \
  --entrypoint python3 \
  ghcr.io/stijndhondt2000/dpp8-9-image:v{version_number} \
  -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.randn(3, device='cuda'))"
```

## 5. Log in to GHCR

Authenticate with a GitHub Personal Access Token that has permission to publish packages.

```bash
sudo docker login ghcr.io -u stijndhondt2000
```

Enter the token when prompted for a password.

## 6. Push the versioned image

```bash
sudo docker push ghcr.io/stijndhondt2000/dpp8-9-image:v{version_number}
```

