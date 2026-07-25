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

To preserve privacy, the app does not preserve a person's face or identity. The source photo will be used only for pose estimation and will not be passed to the diffusion pipeline.

- Python version: 3.10.x

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Naming and rights

This project does not claim affiliation with Gundam, Bandai, or any other rights holder.

