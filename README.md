---
title: GundamPoser
emoji: 🤖
sdk: gradio
python_version: 3.10
app_file: app.py
fullWidth: true
short_description: Transform a human pose into a custom mecha-inspired character.
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

## Naming and rights

This project does not claim affiliation with Gundam, Bandai, or any other
rights holder.
