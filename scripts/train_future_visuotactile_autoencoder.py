"""Train the future-image autoencoder on LIBERO without constructing Pi0.

Example:
    CUDA_VISIBLE_DEVICES=1,2,3,4 PYTHONPATH=src torchrun \
        --standalone --nnodes=1 --nproc_per_node=4 \
        scripts/train_future_visuotactile_autoencoder.py \
        --exp-name libero_headcam_spatial_ae \
        --overwrite
"""

import dataclasses
import logging
import math
import os
import pathlib
import shutil
import time

import imageio.v2 as imageio
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import tyro
import wandb

from openpi.models import pi0_config
from openpi.models_pytorch import future_visuotactile_autoencoder as _autoencoder
from openpi.training import config as _config
from openpi.training import data_loader as _data


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_expert_visuotactile_spatiotemporal_libero"
    exp_name: str = "libero_headcam_spatial_ae"
    output_dir: pathlib.Path = pathlib.Path("checkpoints/future_visuotactile_autoencoder")
    batch_size: int = 256
    num_train_steps: int = 30_000
    num_workers: int = 2
    warmup_steps: int = 500
    peak_lr: float = 1e-4
    end_lr: float = 1e-5
    weight_decay: float = 1e-6
    clip_gradient_norm: float = 1.0
    log_interval: int = 100
    save_interval: int = 1_000
    preview_interval: int = 1_000
    seed: int = 42
    wandb_enabled: bool = False
    overwrite: bool = False
    resume: bool = False


def setup_distributed() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if use_ddp:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return use_ddp, local_rank, device


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def reconstruction_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    model_config: pi0_config.Pi0VisuoTactileConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = _autoencoder.reconstruction_loss_terms(prediction, target)
    total = (
        model_config.future_mse_loss_weight * terms["mse"]
        + model_config.future_charbonnier_loss_weight * terms["charbonnier"]
        + model_config.future_ssim_loss_weight * terms["ssim"]
        + model_config.future_gradient_loss_weight * terms["gradient"]
        + model_config.future_pyramid_loss_weight * terms["pyramid"]
    )
    return total.mean(), terms


def learning_rate(args: Args, step: int) -> float:
    if step < args.warmup_steps:
        return args.peak_lr * (step + 1) / max(1, args.warmup_steps)
    progress = min(1.0, (step - args.warmup_steps) / max(1, args.num_train_steps - args.warmup_steps))
    return args.end_lr + 0.5 * (args.peak_lr - args.end_lr) * (1.0 + math.cos(math.pi * progress))


def checkpoint_root(args: Args) -> pathlib.Path:
    return args.output_dir / args.exp_name


def latest_checkpoint(root: pathlib.Path) -> pathlib.Path | None:
    steps = [int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    return root / str(max(steps)) if steps else None


def save_checkpoint(model, optimizer, args: Args, step: int) -> None:
    root = checkpoint_root(args)
    target = root / str(step)
    temporary = root / f"tmp_{step}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    safetensors.torch.save_model(unwrap(model), temporary / "autoencoder.safetensors")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "args": dataclasses.asdict(args),
        },
        temporary / "training_state.pt",
    )
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)
    logging.info("Saved autoencoder checkpoint to %s", target)


def save_preview(prediction: torch.Tensor, target: torch.Tensor, output_path: pathlib.Path) -> None:
    prediction = prediction[0].detach().float().cpu().numpy()
    target = target[0].detach().float().cpu().numpy()
    if prediction.shape[1] in (1, 3):
        prediction = np.transpose(prediction, (0, 2, 3, 1))
        target = np.transpose(target, (0, 2, 3, 1))
    prediction = np.clip((prediction + 1.0) * 127.5, 0, 255).astype(np.uint8)
    target = np.clip((target + 1.0) * 127.5, 0, 255).astype(np.uint8)
    rows = [np.concatenate(list(frames), axis=1) for frames in (target, prediction)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, np.concatenate(rows, axis=0))


def train(args: Args) -> None:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")
    # LIBERO image transforms use JAX, but this is a PyTorch training process.
    # Keep worker/rank preprocessing on CPU to avoid one CUDA context per worker
    # on every visible GPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    use_ddp, local_rank, device = setup_distributed()
    is_main = not use_ddp or dist.get_rank() == 0
    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    root = checkpoint_root(args)
    if root.exists() and any(root.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"{root} already exists; use --resume or --overwrite.")
    if is_main:
        if root.exists() and args.overwrite:
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    train_config = _config.get_config(args.config_name)
    if not isinstance(train_config.model, pi0_config.Pi0VisuoTactileConfig):
        raise TypeError(f"{args.config_name} does not use Pi0VisuoTactileConfig.")
    if train_config.data.base_config is None:
        raise ValueError("Standalone autoencoder training requires an explicit base data config.")
    autoencoder_data = dataclasses.replace(
        train_config.data,
        base_config=dataclasses.replace(
            train_config.data.base_config,
            # Load only the current head-camera image. No future image sequence
            # is requested or decoded for standalone reconstruction training.
            action_sequence_keys=("actions",),
            action_sequence_extra_steps={},
        ),
    )
    train_config = dataclasses.replace(
        train_config,
        data=autoencoder_data,
        # Only the ignored action field remains sequential, with length one.
        model=dataclasses.replace(train_config.model, action_horizon=1),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = _data.create_data_loader(train_config, framework="pytorch", shuffle=True)

    autoencoder = _autoencoder.create_future_visuotactile_autoencoder(train_config.model).to(device)
    if use_ddp:
        autoencoder = torch.nn.parallel.DistributedDataParallel(
            autoencoder,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    global_step = 0
    if args.resume:
        checkpoint = latest_checkpoint(root)
        if checkpoint is None:
            raise FileNotFoundError(f"No autoencoder checkpoint found in {root}.")
        safetensors.torch.load_model(unwrap(autoencoder), checkpoint / "autoencoder.safetensors", device=str(device))
        state = torch.load(checkpoint / "training_state.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["step"])
        logging.info("Resumed autoencoder training from %s", checkpoint)

    if is_main and args.wandb_enabled:
        wandb.init(project="openpi-future-autoencoder", name=args.exp_name, config=dataclasses.asdict(args))

    autoencoder.train()
    progress = tqdm.tqdm(total=args.num_train_steps, initial=global_step, disable=not is_main)
    recent_metrics = []
    start_time = time.time()
    while global_step < args.num_train_steps:
        for observation, _ in loader:
            if global_step >= args.num_train_steps:
                break
            target = observation.future_visuotactile.to(device)
            lr = learning_rate(args, global_step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction, latents = autoencoder(target)
                loss, terms = reconstruction_objective(prediction, target, train_config.model)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), args.clip_gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics = {
                "loss": float(loss.detach()),
                "mse": float(terms["mse"].mean().detach()),
                "charbonnier": float(terms["charbonnier"].mean().detach()),
                "ssim": float(terms["ssim"].mean().detach()),
                "gradient": float(terms["gradient"].mean().detach()),
                "pyramid": float(terms["pyramid"].mean().detach()),
                "latent_std": float(latents.float().std().detach()),
                "grad_norm": float(grad_norm),
                "lr": lr,
            }
            if device.type == "cuda":
                metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            recent_metrics.append(metrics)
            global_step += 1

            if is_main and global_step % args.log_interval == 0:
                averaged = {key: sum(metric[key] for metric in recent_metrics) / len(recent_metrics) for key in metrics}
                elapsed = time.time() - start_time
                logging.info(
                    "step=%d loss=%.4f mse=%.4f ssim=%.4f latent_std=%.3f "
                    "lr=%.2e grad_norm=%.2f peak_memory=%.2fGiB time=%.1fs",
                    global_step,
                    averaged["loss"],
                    averaged["mse"],
                    averaged["ssim"],
                    averaged["latent_std"],
                    averaged["lr"],
                    averaged["grad_norm"],
                    averaged.get("peak_memory_gib", 0.0),
                    elapsed,
                )
                if args.wandb_enabled:
                    wandb.log(averaged, step=global_step)
                recent_metrics.clear()
                start_time = time.time()

            if is_main and global_step % args.preview_interval == 0:
                save_preview(prediction, target, root / "previews" / f"{global_step}.png")
            if is_main and global_step % args.save_interval == 0:
                save_checkpoint(autoencoder, optimizer, args, global_step)
            if is_main:
                progress.update(1)
                progress.set_postfix(loss=f"{metrics['loss']:.4f}", step=global_step)

    if is_main and global_step % args.save_interval != 0:
        save_checkpoint(autoencoder, optimizer, args, global_step)
    if is_main:
        progress.close()
        if args.wandb_enabled:
            wandb.finish()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    train(tyro.cli(Args))


if __name__ == "__main__":
    main()
