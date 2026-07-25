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

Open the local URL shown in the terminal, then upload a one-person photo or
use the camera input. Choose a scene and select **Generate pose-guided image**.
The app displays the pose overlay, the normalized 384×512 pose map, and one
generated image. The source photo is not sent to the diffusion pipeline.

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

## Naming and rights

This project does not claim affiliation with Gundam, Bandai, or any other
rights holder.
