---
title: GundamPoser
emoji: 🤖
sdk: gradio
sdk_version: 6.20.0
python_version: 3.10.13
app_file: app.py
fullWidth: true
short_description: Turn a human pose into a mecha-inspired character.
models:
  - stable-diffusion-v1-5/stable-diffusion-v1-5
  - lllyasviel/control_v11p_sd15_openpose
  - hw391/AIPI540-GundamPoser-LoRA
tags:
  - diffusion
  - controlnet
  - openpose
  - lora
  - image-generation
---

# GundamPoser

GundamPoser is a hackathon project for:
1. extracting the body pose from an one-person photo and 
2. generating one mecha-inspired person approximately the same pose.

To preserve privacy, the app does not preserve a person's face or identity.
The source photo will be used only for pose estimation and will not be passed
to the diffusion pipeline.

- Python version: 3.10.x

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Run the app:

```bash
.venv/bin/python app.py
```

Open the local URL shown in the terminal, then choose one of the bundled pose
photos, upload a one-person photo, or use the camera input. Optional scene and
seed controls are under **Generation options**. A scene preset fills the full
prompt, which remains editable before generation. Select **Generate** to see
the baseline and trained-LoRA images first; the pose overlay and normalized
384×512 pose map are available under the collapsed pose details. Both images
use the same prompt, seed, and pose. The source photo is not sent to the
diffusion pipeline. Bundled photo sources and licenses are recorded in
`assets/pose-examples/ATTRIBUTION.md`.

Full-body photos with visible arms and legs give ControlNet the strongest pose
signal. Generation uses full-duration OpenPose conditioning and composition
prompts that favor a centered, head-to-toe result.

The body-pose model is downloaded on the first detection request. Stable
Diffusion 1.5 and OpenPose ControlNet are downloaded on the first generation
request. Downloads are cached for later runs. A CUDA GPU is strongly
recommended. On Apple Silicon, the app automatically uses MPS and enables CPU
fallback for operations that MPS does not support. MPS generation uses float32
to avoid half-precision NaNs during image decoding. CPU-only generation can be
very slow.

The automatic device selection order is CUDA, MPS, then CPU. To explicitly
select the local Apple GPU, run:

```bash
GUNDAMPOSER_DEVICE=mps .venv/bin/python app.py
```

## Dataset preparation

Audit image-label pairing, image integrity, and cross-split duplicates without
writing output:

```bash
.venv/bin/python scripts/prepare_dataset.py --dry-run
```

When the audit is clean, prepare deterministic 512×512 train, validation, and
test folders and create a cross-split contact sheet:

```bash
.venv/bin/python scripts/prepare_dataset.py --contact-sheet
```

To intentionally preserve the current source assignments despite cross-split
duplicate or variant images, run:

```bash
.venv/bin/python scripts/prepare_dataset.py \
  --allow-cross-split-duplicates \
  --overwrite \
  --contact-sheet
```

The command refuses a non-empty destination unless `--overwrite` is provided,
and overwrite is restricted to a `data/processed` directory. Source images and
labels are never modified.

## LoRA training

Validate the training configuration and processed training split without
loading model weights:

```bash
.venv/bin/python scripts/train_lora.py --dry-run
```

Run the configured UNet-only LoRA training job:

```bash
.venv/bin/python scripts/train_lora.py
```

The trainer saves resumable adapter checkpoints under `outputs/training`, fixed
prompt validation grids for each checkpoint, and the final adapter at
`outputs/gundamposer_lora.safetensors`. CUDA uses the configured fp16 precision;
Apple Silicon automatically uses float32 for stable MPS training. A local model
snapshot can be supplied with `--base-model /absolute/path/to/snapshot` when
offline.

The comparison app automatically loads the final adapter when it exists. Use a
different checkpoint or adapter with:

```bash
GUNDAMPOSER_LORA_PATH=/absolute/path/to/adapter.safetensors \
  .venv/bin/python app.py
```

## Hugging Face Space deployment

The hosted app uses Gradio with ZeroGPU `large`. Pose extraction runs on CPU;
only the seed-matched baseline and trained diffusion calls use the temporary GPU
allocation. The diffusion pipeline is loaded once at startup as required by
ZeroGPU.

The selected adapter is stored in the private model repository
`hw391/AIPI540-GundamPoser-LoRA`. The protected Space must have an `HF_TOKEN`
secret with read access to that repository. The application uses the secret
only when loading the private LoRA; public base and ControlNet downloads remain
anonymous.

Preview the exact model and Space upload allowlists without making changes:

```bash
.venv/bin/python scripts/deploy_hfspaces.py
```

After reviewing the list, upload the adapter followed by the curated Space
runtime files:

```bash
.venv/bin/python scripts/deploy_hfspaces.py --apply
```

The deployment helper requires local Hugging Face authentication. It does not
read or display token values in its file plan and never uploads `outputs/`
except for the single explicitly selected adapter sent to the private model
repository.

## Naming and rights

This project does not claim affiliation with Gundam, Bandai, or any other
rights holder.
