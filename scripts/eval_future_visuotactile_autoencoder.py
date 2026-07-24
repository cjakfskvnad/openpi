"""Evaluate a standalone future-image autoencoder on LIBERO training images.

Example:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run \
        scripts/eval_future_visuotactile_autoencoder.py \
        --checkpoint-dir \
        checkpoints/future_visuotactile_autoencoder/libero_headcam_singleframe_ae/2000
"""

import csv
import dataclasses
import json
import logging
import math
import os
import pathlib

import numpy as np
from PIL import Image
from PIL import ImageDraw
import safetensors.torch
import torch
import tyro

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_visuotactile_pytorch as _visuotactile
from openpi.training import config as _config
from openpi.training import data_loader as _data


@dataclasses.dataclass
class Args:
    checkpoint_dir: pathlib.Path
    config_name: str = "pi05_visuotactile_libero"
    output_dir: pathlib.Path = pathlib.Path("data/libero/autoencoder_reconstruction")
    num_samples: int = 32
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    device: str | None = None
    contact_sheet_samples: int = 12


def _standalone_config(args: Args) -> _config.TrainConfig:
    train_config = _config.get_config(args.config_name)
    if not isinstance(train_config.model, pi0_config.Pi0VisuoTactileConfig):
        raise TypeError(f"{args.config_name} does not use Pi0VisuoTactileConfig.")
    if train_config.data.base_config is None:
        raise ValueError("Evaluation requires an explicit base data config.")
    data = dataclasses.replace(
        train_config.data,
        base_config=dataclasses.replace(
            train_config.data.base_config,
            action_sequence_keys=("actions",),
            action_sequence_extra_steps={},
        ),
    )
    return dataclasses.replace(
        train_config,
        data=data,
        model=dataclasses.replace(train_config.model, action_horizon=1),
        batch_size=min(args.batch_size, args.num_samples),
        num_workers=args.num_workers,
        seed=args.seed,
    )


def _to_uint8(images: torch.Tensor) -> np.ndarray:
    return (
        images.detach()
        .float()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def _metric_rows(prediction: torch.Tensor, target: torch.Tensor) -> list[dict[str, float | int]]:
    terms = _visuotactile._reconstruction_loss_terms(prediction, target)  # noqa: SLF001
    mse_normalized = terms["mse"].flatten()
    mse_01 = mse_normalized / 4.0
    mae_01 = (prediction.float() - target.float()).abs().flatten(2).mean(2).flatten() / 2.0
    rows = []
    for index in range(len(mse_01)):
        mse = float(mse_01[index])
        rows.append(
            {
                "sample": index,
                "mse": mse,
                "mae": float(mae_01[index]),
                "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
                "ms_ssim": float(1.0 - terms["ssim"].flatten()[index]),
            }
        )
    return rows


def _save_comparison(path: pathlib.Path, target: np.ndarray, prediction: np.ndarray) -> None:
    error = np.abs(target.astype(np.int16) - prediction.astype(np.int16)).astype(np.uint8)
    error = np.clip(error.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    canvas = Image.new("RGB", (target.shape[1] * 3, target.shape[0] + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, image) in enumerate(
        (("target", target), ("reconstruction", prediction), ("|error| x4", error))
    ):
        x = column * target.shape[1]
        canvas.paste(Image.fromarray(image), (x, 24))
        draw.text((x + 4, 5), label, fill="black")
    canvas.save(path)


def _save_contact_sheet(
    path: pathlib.Path,
    targets: np.ndarray,
    predictions: np.ndarray,
    rows: list[dict[str, float | int]],
    count: int,
) -> None:
    count = min(count, len(targets))
    width = targets.shape[2]
    height = targets.shape[1]
    header_height = 28
    row_height = height + 22
    canvas = Image.new("RGB", (2 * width, header_height + count * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 7), "target", fill="black")
    draw.text((width + 4, 7), "reconstruction", fill="black")
    for index in range(count):
        y = header_height + index * row_height
        canvas.paste(Image.fromarray(targets[index]), (0, y))
        canvas.paste(Image.fromarray(predictions[index]), (width, y))
        metrics = rows[index]
        draw.text(
            (4, y + height + 3),
            f"sample {index:03d}   PSNR {metrics['psnr_db']:.2f} dB   MS-SSIM {metrics['ms_ssim']:.4f}",
            fill="black",
        )
    canvas.save(path)


def main(args: Args) -> None:
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive.")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    config = _standalone_config(args)
    loader = _data.create_data_loader(
        config,
        framework="pytorch",
        shuffle=True,
        num_batches=math.ceil(args.num_samples / config.batch_size),
    )
    model = _visuotactile.create_future_visuotactile_autoencoder(config.model).to(device)
    weight_path = args.checkpoint_dir / "autoencoder.safetensors"
    safetensors.torch.load_model(model, weight_path, device=str(device))
    model.eval()

    all_targets = []
    all_predictions = []
    with torch.inference_mode():
        for observation, _ in loader:
            target = observation.future_visuotactile.to(device)
            prediction, _ = model(target)
            all_targets.append(target.cpu())
            all_predictions.append(prediction.cpu())

    target = torch.cat(all_targets)[: args.num_samples]
    prediction = torch.cat(all_predictions)[: args.num_samples]
    rows = _metric_rows(prediction, target)
    target_images = _to_uint8(target[:, 0])
    prediction_images = _to_uint8(prediction[:, 0])

    output_dir = args.output_dir / f"{args.checkpoint_dir.parent.name}_{args.checkpoint_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (target_image, prediction_image) in enumerate(zip(target_images, prediction_images, strict=True)):
        _save_comparison(output_dir / f"sample_{index:03d}.png", target_image, prediction_image)
    _save_contact_sheet(
        output_dir / "contact_sheet.png",
        target_images,
        prediction_images,
        rows,
        args.contact_sheet_samples,
    )

    with (output_dir / "per_sample_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint_dir.resolve()),
        "config_name": args.config_name,
        "dataset": config.data.repo_id,
        "split": "train",
        "sampling": f"seeded shuffle (seed={args.seed})",
        "num_samples": len(rows),
        "metrics": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows])),
                "min": float(np.min([row[key] for row in rows])),
                "max": float(np.max([row[key] for row in rows])),
            }
            for key in ("mse", "mae", "psnr_db", "ms_ssim")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("Evaluated %d training images on %s", len(rows), device)
    logging.info("PSNR: %.2f dB", summary["metrics"]["psnr_db"]["mean"])
    logging.info("MS-SSIM: %.4f", summary["metrics"]["ms_ssim"]["mean"])
    logging.info("Saved results to %s", output_dir.resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
