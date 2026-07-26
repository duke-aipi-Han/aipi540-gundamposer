#!/usr/bin/env python3
"""Train and validate a UNet-only Stable Diffusion LoRA adapter."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from typing import Any, Iterator, Sequence

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
)
from peft import LoraConfig as PeftLoraConfig
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPTextModel

from gundamposer.config import ConfigError, TrainingConfig, load_training_config
from gundamposer.preprocessing import to_oriented_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
DEFAULT_TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "train"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
DEFAULT_FINAL_ADAPTER = PROJECT_ROOT / "outputs" / "gundamposer_lora.safetensors"
ADAPTER_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
ADAPTER_NAME = "gundamposer"


class TrainingError(ValueError):
    """Raised when training inputs or runtime settings are invalid."""


@dataclass(frozen=True)
class TrainingOverrides:
    max_train_steps: int | None = None
    checkpointing_steps: int | None = None


@dataclass(frozen=True)
class EffectiveTrainingSettings:
    device: str
    mixed_precision: str
    weight_dtype: torch.dtype
    max_train_steps: int
    checkpointing_steps: int


@dataclass(frozen=True)
class TrainingSummary:
    image_count: int
    max_train_steps: int
    checkpointing_steps: int
    effective_batch_size: int
    device: str
    mixed_precision: str


class TrainingImageDataset(Dataset[dict[str, torch.Tensor]]):
    """Load one ImageFolder-style split and tokenize its fixed captions."""

    def __init__(
        self,
        root: Path,
        tokenizer: Any,
        *,
        resolution: int,
    ) -> None:
        metadata_path = root / "metadata.jsonl"
        if not metadata_path.is_file():
            raise TrainingError(f"Training metadata does not exist: {metadata_path}")
        rows: list[tuple[Path, str]] = []
        for line_number, line in enumerate(
            metadata_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                row = json.loads(line)
                image_path = root / row["file_name"]
                caption = row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise TrainingError(
                    f"Invalid metadata row {line_number}: {metadata_path}"
                ) from error
            if not image_path.is_file():
                raise TrainingError(f"Metadata image does not exist: {image_path}")
            if not isinstance(caption, str) or not caption.strip():
                raise TrainingError(
                    f"Metadata caption must be non-empty at row {line_number}."
                )
            rows.append((image_path, caption.strip()))
        if not rows:
            raise TrainingError("The training split contains no metadata rows.")

        self._rows = tuple(rows)
        self._tokenizer = tokenizer
        self._resolution = resolution

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, caption = self._rows[index]
        with Image.open(image_path) as image:
            normalized = to_oriented_rgb(image)
            if normalized.size != (self._resolution, self._resolution):
                raise TrainingError(
                    f"Training image must be {self._resolution}x{self._resolution}: "
                    f"{image_path}"
                )
            image_tensor = torch.from_numpy(
                np.asarray(normalized, dtype=np.float32).copy()
            )
        pixel_values = image_tensor.permute(2, 0, 1).div(127.5).sub(1.0)
        tokenized = self._tokenizer(
            caption,
            max_length=self._tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def resolve_training_device(requested: str | None = None) -> str:
    """Resolve CUDA, MPS, or CPU without silently accepting an unavailable device."""

    if requested is not None:
        if requested not in {"cuda", "mps", "cpu"}:
            raise TrainingError("device must be one of: cuda, mps, cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise TrainingError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise TrainingError("MPS was requested but is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def effective_training_settings(
    config: TrainingConfig,
    *,
    device: str,
    overrides: TrainingOverrides | None = None,
) -> EffectiveTrainingSettings:
    """Apply safe backend precision and positive CLI step overrides."""

    overrides = overrides or TrainingOverrides()
    max_train_steps = overrides.max_train_steps or config.training.max_train_steps
    checkpointing_steps = (
        overrides.checkpointing_steps or config.training.checkpointing_steps
    )
    if max_train_steps <= 0 or checkpointing_steps <= 0:
        raise TrainingError("training and checkpoint steps must be positive")
    if checkpointing_steps > max_train_steps:
        raise TrainingError("checkpointing steps cannot exceed training steps")

    mixed_precision = config.training.mixed_precision if device == "cuda" else "no"
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    return EffectiveTrainingSettings(
        device=device,
        mixed_precision=mixed_precision,
        weight_dtype=weight_dtype,
        max_train_steps=max_train_steps,
        checkpointing_steps=checkpointing_steps,
    )


def _infinite_batches(dataloader: DataLoader[Any]) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from dataloader


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise TrainingError(f"Invalid checkpoint directory name: {path.name}") from error


def resolve_resume_checkpoint(output_dir: Path, requested: str | None) -> Path | None:
    """Resolve an explicit checkpoint or the latest checkpoint in one run."""

    if requested is None:
        return None
    if requested == "latest":
        checkpoints = sorted(
            (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
            key=_checkpoint_step,
        )
        if not checkpoints:
            raise TrainingError(f"No checkpoints exist in {output_dir}")
        return checkpoints[-1]
    checkpoint = Path(requested).resolve()
    if not checkpoint.is_dir():
        raise TrainingError(f"Resume checkpoint does not exist: {checkpoint}")
    _checkpoint_step(checkpoint)
    return checkpoint


def _lora_state_dict(unet: UNet2DConditionModel) -> dict[str, torch.Tensor]:
    return convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))


def _save_lora(unet: UNet2DConditionModel, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=destination.parent,
        unet_lora_layers=_lora_state_dict(unet),
        weight_name=destination.name,
        safe_serialization=True,
    )


def _load_lora(unet: UNet2DConditionModel, checkpoint: Path) -> None:
    state_dict, _ = LoraLoaderMixin.lora_state_dict(checkpoint)
    unet_state = {
        key.removeprefix("unet."): value
        for key, value in state_dict.items()
        if key.startswith("unet.")
    }
    peft_state = convert_unet_state_dict_to_peft(unet_state)
    result = set_peft_model_state_dict(unet, peft_state, adapter_name="default")
    if getattr(result, "unexpected_keys", None):
        raise TrainingError(
            f"Unexpected adapter keys in checkpoint: {result.unexpected_keys}"
        )


def _save_checkpoint(
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    *,
    output_dir: Path,
    global_step: int,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{global_step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    _save_lora(unet, checkpoint / ADAPTER_WEIGHT_NAME)
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        checkpoint / "training_state.pt",
    )
    return checkpoint


def _load_checkpoint_state(
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
) -> int:
    _load_lora(unet, checkpoint)
    state_path = checkpoint / "training_state.pt"
    if not state_path.is_file():
        raise TrainingError(f"Checkpoint state does not exist: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    global_step = int(state["global_step"])
    if global_step != _checkpoint_step(checkpoint):
        raise TrainingError("Checkpoint state and directory step do not match")
    return global_step


def _write_run_snapshots(
    config_path: Path,
    output_dir: Path,
    settings: EffectiveTrainingSettings,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "training.yaml")
    packages = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "peft"):
        packages[name] = importlib.metadata.version(name)
    snapshot = {
        "python": sys.version,
        "platform": sys.platform,
        "settings": {
            **asdict(settings),
            "weight_dtype": str(settings.weight_dtype),
        },
        "packages": packages,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inspect_training_run(
    config: TrainingConfig,
    train_data: Path,
    *,
    device: str,
    overrides: TrainingOverrides | None = None,
) -> TrainingSummary:
    settings = effective_training_settings(
        config,
        device=device,
        overrides=overrides,
    )
    metadata_path = train_data / "metadata.jsonl"
    if not metadata_path.is_file():
        raise TrainingError(f"Training metadata does not exist: {metadata_path}")
    rows = metadata_path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise TrainingError("The training split contains no metadata rows.")
    for line in rows:
        row = json.loads(line)
        image_path = train_data / row["file_name"]
        if not image_path.is_file():
            raise TrainingError(f"Metadata image does not exist: {image_path}")
    return TrainingSummary(
        image_count=len(rows),
        max_train_steps=settings.max_train_steps,
        checkpointing_steps=settings.checkpointing_steps,
        effective_batch_size=(
            config.training.train_batch_size
            * config.training.gradient_accumulation_steps
        ),
        device=device,
        mixed_precision=settings.mixed_precision,
    )


def train(
    config: TrainingConfig,
    *,
    config_path: Path,
    train_data: Path,
    output_dir: Path,
    final_adapter: Path,
    device: str,
    model_source: str | Path | None = None,
    overrides: TrainingOverrides | None = None,
    resume_from_checkpoint: str | None = None,
) -> tuple[list[Path], float]:
    """Train the adapter and return created checkpoints plus elapsed seconds."""

    settings = effective_training_settings(
        config,
        device=device,
        overrides=overrides,
    )
    pretrained_source = str(model_source or config.base_model)
    accelerator = Accelerator(
        mixed_precision=settings.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        cpu=device == "cpu",
    )
    if accelerator.device.type != device:
        raise TrainingError(
            f"Accelerate selected {accelerator.device.type}, expected {device}."
        )
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_source,
        subfolder="tokenizer",
        use_fast=False,
        token=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_source,
        subfolder="scheduler",
        token=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_source,
        subfolder="text_encoder",
        torch_dtype=settings.weight_dtype,
        token=False,
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_source,
        subfolder="vae",
        torch_dtype=settings.weight_dtype,
        token=False,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_source,
        subfolder="unet",
        torch_dtype=torch.float32,
        token=False,
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.add_adapter(
        PeftLoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    if config.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    trainable_parameters = [
        parameter for parameter in unet.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise TrainingError("No trainable LoRA parameters were created.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
    )
    resume_checkpoint = resolve_resume_checkpoint(output_dir, resume_from_checkpoint)
    global_step = 0
    if resume_checkpoint is not None:
        global_step = _load_checkpoint_state(unet, optimizer, resume_checkpoint)
        if global_step >= settings.max_train_steps:
            raise TrainingError(
                "The resume checkpoint is already at or beyond max_train_steps."
            )

    dataset = TrainingImageDataset(
        train_data,
        tokenizer,
        resolution=config.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device, dtype=settings.weight_dtype)
    text_encoder.to(accelerator.device, dtype=settings.weight_dtype)
    vae.eval()
    text_encoder.eval()
    unet.train()
    if accelerator.is_main_process:
        _write_run_snapshots(config_path, output_dir, settings)

    started = perf_counter()
    checkpoints: list[Path] = []
    batches = _infinite_batches(dataloader)
    while global_step < settings.max_train_steps:
        batch = next(batches)
        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(
                accelerator.device,
                dtype=settings.weight_dtype,
            )
            input_ids = batch["input_ids"].to(accelerator.device)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            prediction = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                return_dict=False,
            )[0]
            prediction_type = noise_scheduler.config.prediction_type
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise TrainingError(
                    f"Unsupported scheduler prediction type: {prediction_type}"
                )
            loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            if accelerator.is_main_process:
                print(
                    f"step={global_step}/{settings.max_train_steps} "
                    f"loss={loss.detach().item():.6f}",
                    flush=True,
                )
            if global_step % settings.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(unet)
                    checkpoints.append(
                        _save_checkpoint(
                            unwrapped,
                            optimizer,
                            output_dir=output_dir,
                            global_step=global_step,
                        )
                    )

    accelerator.wait_for_everyone()
    elapsed = perf_counter() - started
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        if not checkpoints or _checkpoint_step(checkpoints[-1]) != global_step:
            checkpoints.append(
                _save_checkpoint(
                    unwrapped,
                    optimizer,
                    output_dir=output_dir,
                    global_step=global_step,
                )
            )
        _save_lora(unwrapped, final_adapter)
        (output_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "elapsed_seconds": elapsed,
                    "seconds_per_step": elapsed / max(1, global_step),
                    "final_adapter": str(final_adapter.resolve()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.end_training()
    del unet, optimizer, dataloader, vae, text_encoder
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return checkpoints, elapsed


def _validation_grid(images: Sequence[Image.Image], labels: Sequence[str]) -> Image.Image:
    width = 512
    label_height = 48
    grid = Image.new("RGB", (width * len(images), 512 + label_height), "white")
    draw = ImageDraw.Draw(grid)
    for index, (image, label) in enumerate(zip(images, labels)):
        grid.paste(image.resize((512, 512)), (index * width, label_height))
        draw.text((index * width + 8, 14), label, fill="black")
    return grid


def validate_checkpoints(
    config: TrainingConfig,
    checkpoints: Sequence[Path],
    *,
    output_dir: Path,
    device: str,
    model_source: str | Path | None = None,
) -> None:
    """Generate fixed prompt/seed images for every saved checkpoint."""

    if not checkpoints:
        return
    dtype = torch.float16 if device == "cuda" else torch.float32
    pretrained_source = str(model_source or config.base_model)
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_source,
        torch_dtype=dtype,
        use_safetensors=True,
        token=False,
    )
    if device == "mps":
        pipeline.enable_attention_slicing()
    pipeline.to(device)
    validation_root = output_dir / "validation"
    for checkpoint in checkpoints:
        pipeline.unload_lora_weights()
        pipeline.load_lora_weights(checkpoint, adapter_name=ADAPTER_NAME)
        pipeline.set_adapters(ADAPTER_NAME, adapter_weights=1.0)
        images: list[Image.Image] = []
        labels: list[str] = []
        checkpoint_root = validation_root / checkpoint.name
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for index, prompt in enumerate(config.validation.prompts, start=1):
            generator_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(
                config.validation.seed
            )
            autocast = (
                torch.autocast("cuda") if device == "cuda" else nullcontext()
            )
            with torch.inference_mode(), autocast:
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=24,
                    guidance_scale=7.0,
                    generator=generator,
                    width=512,
                    height=512,
                )
            image = result.images[0]
            image.save(checkpoint_root / f"prompt_{index:02d}.png")
            images.append(image)
            labels.append(f"prompt {index} · seed {config.validation.seed}")
        _validation_grid(images, labels).save(checkpoint_root / "grid.png")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Optional local model snapshot to use instead of the configured Hub ID.",
    )
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-adapter", type=Path, default=DEFAULT_FINAL_ADAPTER)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--checkpointing-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        config = load_training_config(args.config)
        device = resolve_training_device(args.device)
        overrides = TrainingOverrides(
            max_train_steps=args.max_train_steps,
            checkpointing_steps=args.checkpointing_steps,
        )
        summary = inspect_training_run(
            config,
            args.train_data.resolve(),
            device=device,
            overrides=overrides,
        )
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        if args.dry_run:
            return 0

        if args.validate_only:
            checkpoints = sorted(
                args.output_dir.resolve().glob("checkpoint-*"),
                key=_checkpoint_step,
            )
            if not checkpoints:
                raise TrainingError("No checkpoints are available for validation.")
            validate_checkpoints(
                config,
                checkpoints,
                output_dir=args.output_dir.resolve(),
                device=device,
                model_source=args.base_model,
            )
            return 0

        checkpoints, elapsed = train(
            config,
            config_path=args.config.resolve(),
            train_data=args.train_data.resolve(),
            output_dir=args.output_dir.resolve(),
            final_adapter=args.final_adapter.resolve(),
            device=device,
            model_source=args.base_model,
            overrides=overrides,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        print(f"Training completed in {elapsed:.1f}s.")
        if not args.skip_validation:
            validate_checkpoints(
                config,
                checkpoints,
                output_dir=args.output_dir.resolve(),
                device=device,
                model_source=args.base_model,
            )
    except (ConfigError, TrainingError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
