"""PyTorch pi0 variant with visuo-tactile prefix tokens and future prediction.

This module intentionally mirrors ``pi0_pytorch.PI0Pytorch`` so existing pi0
checkpoints can be partially loaded while new visuo-tactile and future heads
remain randomly initialized.
"""

from collections.abc import Iterable
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


def _config_get(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


class ConvResidualBlock(nn.Module):
    """Small residual block used by the spatial future-image decoder."""

    def __init__(self, channels: int):
        super().__init__()
        num_groups = min(32, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class FutureVisuoTactileVisionAutoencoder(nn.Module):
    """Self-contained convolutional autoencoder for future images.

    This module intentionally owns both its encoder and decoder. It has no
    dependency on Pi0, PaliGemma, or SigLIP, so it can be pretrained directly on
    image sequences and later loaded as a frozen latent codec by the policy.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        future_shape: tuple[int, ...],
        encoder_width: int,
        decoder_width: int,
        decoder_depth: int,
        latent_grid_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        if len(future_shape) != 3:
            raise ValueError(
                "future_visuotactile_shape must be image-like with 3 dims, e.g. (3, H, W) or (H, W, 3), "
                f"got {future_shape}."
            )
        if future_shape[0] in (1, 3):
            channels, height, width = future_shape
            self.channels_first = True
        elif future_shape[-1] in (1, 3):
            height, width, channels = future_shape
            self.channels_first = False
        else:
            raise ValueError(
                "future_visuotactile_shape must have 1 or 3 channels in the first or last dimension, "
                f"got {future_shape}."
            )

        self.latent_channels = latent_dim
        self.future_shape = future_shape
        self.out_channels = channels
        self.out_height = height
        self.out_width = width

        if latent_grid_size is None:
            latent_grid_size = (14, 14)
        self.latent_grid_h, self.latent_grid_w = latent_grid_size
        if self.latent_grid_h <= 0 or self.latent_grid_w <= 0:
            raise ValueError(f"latent_grid_size must be positive, got {latent_grid_size}.")
        height_ratio = height / self.latent_grid_h
        width_ratio = width / self.latent_grid_w
        if height_ratio != width_ratio or not height_ratio.is_integer():
            raise ValueError(
                "Image size must be an equal integer multiple of latent_grid_size, "
                f"got image {(height, width)} and grid {latent_grid_size}."
            )
        num_downsamples = int(math.log2(int(height_ratio)))
        if 2**num_downsamples != int(height_ratio):
            raise ValueError(
                "Image-to-latent ratio must be a power of two, "
                f"got image {(height, width)} and grid {latent_grid_size}."
            )
        self.flat_latent_dim = self.latent_channels * self.latent_grid_h * self.latent_grid_w

        encoder_stages = []
        current_width = max(32, encoder_width)
        self.encoder_in = nn.Conv2d(channels, current_width, kernel_size=3, padding=1)
        for _ in range(num_downsamples):
            next_width = min(max(64, decoder_width), current_width * 2)
            encoder_stages.append(
                nn.Sequential(
                    ConvResidualBlock(current_width),
                    nn.Conv2d(current_width, next_width, kernel_size=4, stride=2, padding=1),
                )
            )
            current_width = next_width
        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.encoder_out = nn.Sequential(
            ConvResidualBlock(current_width),
            nn.GroupNorm(min(32, current_width), current_width),
            nn.SiLU(),
            nn.Conv2d(current_width, self.latent_channels, kernel_size=3, padding=1),
        )

        bottleneck_width = max(64, decoder_width)
        self.decoder_in = nn.Conv2d(self.latent_channels, bottleneck_width, kernel_size=3, padding=1)
        self.decoder_bottleneck = nn.Sequential(
            *(ConvResidualBlock(bottleneck_width) for _ in range(max(1, decoder_depth)))
        )

        decoder_stages = []
        current_width = bottleneck_width
        for _ in range(num_downsamples):
            next_width = max(32, current_width // 2)
            decoder_stages.append(
                nn.ModuleDict(
                    {
                        "residual": ConvResidualBlock(current_width),
                        "projection": nn.Conv2d(current_width, next_width, kernel_size=3, padding=1),
                    }
                )
            )
            current_width = next_width
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.decoder_out = nn.Sequential(
            ConvResidualBlock(current_width),
            nn.GroupNorm(min(32, current_width), current_width),
            nn.SiLU(),
            nn.Conv2d(current_width, channels, kernel_size=3, padding=1),
        )

    def _future_to_image_batch(self, future_visuotactile: Tensor) -> tuple[Tensor, int, int]:
        if future_visuotactile.ndim != 5:
            raise ValueError(
                "future_visuotactile must have shape [batch, horizon, C, H, W] or [batch, horizon, H, W, C], "
                f"got {tuple(future_visuotactile.shape)}."
            )
        batch_size, horizon = future_visuotactile.shape[:2]
        if self.channels_first:
            images = future_visuotactile.reshape(batch_size * horizon, *future_visuotactile.shape[2:])
        else:
            images = future_visuotactile.permute(0, 1, 4, 2, 3).reshape(
                batch_size * horizon, future_visuotactile.shape[-1], *future_visuotactile.shape[2:4]
            )
        return images, batch_size, horizon

    def encode(self, future_visuotactile: Tensor) -> Tensor:
        images, batch_size, horizon = self._future_to_image_batch(future_visuotactile)
        if images.shape[-2:] != (self.out_height, self.out_width):
            images = F.interpolate(images, size=(self.out_height, self.out_width), mode="bilinear", align_corners=False)
        x = self.encoder_in(images.to(dtype=torch.float32))
        for stage in self.encoder_stages:
            x = stage(x)
        latents = self.encoder_out(x)
        if latents.shape[-2:] != (self.latent_grid_h, self.latent_grid_w):
            raise RuntimeError(
                f"Encoder produced grid {tuple(latents.shape[-2:])}, expected "
                f"{(self.latent_grid_h, self.latent_grid_w)}."
            )
        latents = latents.flatten(1)
        # A fixed per-frame normalization gives flow matching a stable target
        # scale and prevents the encoder from winning by shrinking its codes.
        latents = F.layer_norm(latents, (self.flat_latent_dim,))
        return latents.reshape(batch_size, horizon, self.flat_latent_dim)

    def forward(self, future_visuotactile: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(future_visuotactile)
        return self.decode(latents), latents

    def decode(self, future_latents: Tensor) -> Tensor:
        batch_size, horizon = future_latents.shape[:2]
        if future_latents.shape[-1] != self.flat_latent_dim:
            raise ValueError(f"Expected future latent dim {self.flat_latent_dim}, got {future_latents.shape[-1]}.")
        x = future_latents.reshape(
            batch_size * horizon,
            self.latent_channels,
            self.latent_grid_h,
            self.latent_grid_w,
        )
        x = self.decoder_bottleneck(self.decoder_in(x))
        for stage in self.decoder_stages:
            x = stage["residual"](x)
            next_height = min(self.out_height, x.shape[-2] * 2)
            next_width = min(self.out_width, x.shape[-1] * 2)
            x = F.interpolate(x, size=(next_height, next_width), mode="bilinear", align_corners=False)
            x = stage["projection"](x)
        if x.shape[-2:] != (self.out_height, self.out_width):
            x = F.interpolate(x, size=(self.out_height, self.out_width), mode="bilinear", align_corners=False)
        image = torch.tanh(self.decoder_out(x))
        if self.channels_first:
            return image.reshape(batch_size, horizon, self.out_channels, self.out_height, self.out_width)
        return image.reshape(batch_size, horizon, self.out_channels, self.out_height, self.out_width).permute(
            0, 1, 3, 4, 2
        )


def _future_images_to_nchw(images: Tensor) -> Tensor:
    """Convert [B, T, H, W, C] or [B, T, C, H, W] images to [B*T, C, H, W]."""
    if images.ndim != 5:
        raise ValueError(f"Expected rank-5 future images, got {tuple(images.shape)}.")
    batch_size, horizon = images.shape[:2]
    if images.shape[2] in (1, 3):
        return images.reshape(batch_size * horizon, *images.shape[2:])
    if images.shape[-1] in (1, 3):
        return images.permute(0, 1, 4, 2, 3).reshape(
            batch_size * horizon,
            images.shape[-1],
            images.shape[2],
            images.shape[3],
        )
    raise ValueError(f"Could not infer the channel dimension for future images shaped {tuple(images.shape)}.")


def _ssim_loss(pred: Tensor, target: Tensor, window_size: int = 7, levels: int = 4) -> Tensor:
    """Dependency-free multi-scale SSIM loss, returned per image."""
    pred = (pred + 1.0) * 0.5
    target = (target + 1.0) * 0.5
    losses = []
    for _ in range(levels):
        padding = window_size // 2
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
        sigma_pred = (F.avg_pool2d(pred.square(), window_size, stride=1, padding=padding) - mu_pred.square()).clamp_min(
            0.0
        )
        sigma_target = (
            F.avg_pool2d(target.square(), window_size, stride=1, padding=padding) - mu_target.square()
        ).clamp_min(0.0)
        sigma_cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_pred * mu_target
        c1 = 0.01**2
        c2 = 0.03**2
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / (
            (mu_pred.square() + mu_target.square() + c1) * (sigma_pred + sigma_target + c2)
        ).clamp_min(1e-6)
        losses.append((1.0 - ssim.clamp(-1.0, 1.0)).flatten(1).mean(1))
        if min(pred.shape[-2:]) <= window_size * 2:
            break
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target = F.avg_pool2d(target, kernel_size=2, stride=2)
    return torch.stack(losses).mean(0)


def _gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = (pred_dx - target_dx).abs().flatten(1).mean(1)
    loss_y = (pred_dy - target_dy).abs().flatten(1).mean(1)
    return 0.5 * (loss_x + loss_y)


def _pyramid_loss(pred: Tensor, target: Tensor, levels: int = 4) -> Tensor:
    """Multi-scale structural L1 loss used as a lightweight perceptual term."""
    losses = []
    for _ in range(levels):
        losses.append((pred - target).abs().flatten(1).mean(1))
        if min(pred.shape[-2:]) <= 8:
            break
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target = F.avg_pool2d(target, kernel_size=2, stride=2)
    return torch.stack(losses).mean(0)


def _reconstruction_loss_terms(pred: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Return per-[batch, horizon] image reconstruction terms."""
    batch_size, horizon = pred.shape[:2]
    pred_nchw = _future_images_to_nchw(pred).to(dtype=torch.float32)
    target_nchw = _future_images_to_nchw(target).to(dtype=torch.float32)
    difference = pred_nchw - target_nchw
    reshape = lambda value: value.reshape(batch_size, horizon)  # noqa: E731
    return {
        "mse": reshape(difference.square().flatten(1).mean(1)),
        "charbonnier": reshape(torch.sqrt(difference.square() + 1e-6).flatten(1).mean(1)),
        "ssim": reshape(_ssim_loss(pred_nchw, target_nchw)),
        "gradient": reshape(_gradient_loss(pred_nchw, target_nchw)),
        "pyramid": reshape(_pyramid_loss(pred_nchw, target_nchw)),
    }


def _temporal_difference_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Match frame-to-frame motion and return a [batch, horizon] loss."""
    pred_nchw = _future_images_to_nchw(pred)
    target_nchw = _future_images_to_nchw(target)
    batch_size, horizon = pred.shape[:2]
    pred_nchw = pred_nchw.reshape(batch_size, horizon, *pred_nchw.shape[1:])
    target_nchw = target_nchw.reshape(batch_size, horizon, *target_nchw.shape[1:])
    temporal = (pred_nchw[:, 1:] - pred_nchw[:, :-1]) - (target_nchw[:, 1:] - target_nchw[:, :-1])
    temporal = temporal.to(dtype=torch.float32).abs().flatten(2).mean(2)
    return F.pad(temporal, (1, 0))


def create_future_visuotactile_autoencoder(config) -> FutureVisuoTactileVisionAutoencoder:
    """Build the standalone future-image codec from a Pi0 visuo-tactile config."""
    future_shape = _config_get(config, "future_visuotactile_shape", None)
    if future_shape is None:
        raise ValueError("future_visuotactile_shape is required to create the autoencoder.")
    return FutureVisuoTactileVisionAutoencoder(
        latent_dim=_config_get(config, "future_visuotactile_latent_dim", 16),
        future_shape=tuple(future_shape),
        encoder_width=_config_get(config, "future_visuotactile_encoder_width", 64),
        decoder_width=_config_get(config, "future_visuotactile_decoder_width", 256),
        decoder_depth=_config_get(config, "future_visuotactile_decoder_depth", 2),
        latent_grid_size=_config_get(config, "future_visuotactile_latent_grid_size", None),
    )


class PI0VisuoTactilePytorch(PI0Pytorch):
    """pi0 PyTorch model with current and future visuo-tactile streams.

    Expected optional config fields:
    - ``visuotactile_keys``: keys to read from ``observation.images`` as current
      visuo-tactile image-like inputs. Defaults to common tactile names.
    - ``future_visuotactile_key``: field/key used when future target is stored in
      the observation. Defaults to ``"future_visuotactile"``.
    - ``future_visuotactile_shape``: image-like tail shape of future target.
    - ``future_visuotactile_latent_dim``: latent dim concatenated with action.
    - ``action_loss_weight``: action flow loss multiplier.
    - ``future_flow_loss_weight``: future latent flow loss multiplier.
    - ``future_visuotactile_loss_weight``: future reconstruction loss multiplier.
    """

    def __init__(self, config):
        super().__init__(config)

        expert_width = self.action_in_proj.out_features

        self.visuotactile_keys = _as_tuple(
            _config_get(config, "visuotactile_keys", ("visuotactile_0_rgb", "tactile_0_rgb"))
        )
        self.future_visuotactile_key = _config_get(config, "future_visuotactile_key", "future_visuotactile")
        self.action_loss_weight = _config_get(config, "action_loss_weight", 1.0)
        self.future_flow_loss_weight = _config_get(config, "future_flow_loss_weight", 1.0)
        self.future_visuotactile_loss_weight = _config_get(config, "future_visuotactile_loss_weight", 1.0)
        self.future_autoencoder_loss_weight = _config_get(config, "future_autoencoder_loss_weight", 1.0)
        self.future_mse_loss_weight = _config_get(config, "future_mse_loss_weight", 0.1)
        self.future_charbonnier_loss_weight = _config_get(config, "future_charbonnier_loss_weight", 1.0)
        self.future_ssim_loss_weight = _config_get(config, "future_ssim_loss_weight", 0.2)
        self.future_gradient_loss_weight = _config_get(config, "future_gradient_loss_weight", 0.1)
        self.future_pyramid_loss_weight = _config_get(config, "future_pyramid_loss_weight", 0.1)
        self.future_temporal_loss_weight = _config_get(config, "future_temporal_loss_weight", 0.2)

        self.future_visuotactile_autoencoder = create_future_visuotactile_autoencoder(config)
        self.pretrained_autoencoder_loaded = False
        self.future_visuotactile_latent_dim = self.future_visuotactile_autoencoder.flat_latent_dim

        self.joint_action_dim = config.action_dim + self.future_visuotactile_latent_dim
        self.action_future_in_proj = nn.Linear(self.joint_action_dim, expert_width)
        self.action_future_out_proj = nn.Linear(expert_width, self.joint_action_dim)

        if not self.pi05:
            self.action_future_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
            self.action_future_time_mlp_out = nn.Linear(expert_width, expert_width)

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()

    def get_training_phase(self, step: int) -> str:
        autoencoder_steps = _config_get(self.config, "future_autoencoder_pretrain_steps", 0)
        joint_start_step = _config_get(self.config, "future_joint_finetune_start_step", autoencoder_steps)
        if not self.pretrained_autoencoder_loaded and step < autoencoder_steps:
            return "autoencoder"
        if step < joint_start_step:
            return "flow"
        return "joint"

    def set_train_phase(self, phase: str) -> None:
        """Select trainable modules for staged future-image training."""
        if phase not in {"autoencoder", "flow", "joint"}:
            raise ValueError(f"Unknown visuo-tactile training phase: {phase}.")
        for parameter in self.parameters():
            parameter.requires_grad_(phase == "joint")

        if phase == "autoencoder":
            for parameter in self.future_visuotactile_autoencoder.parameters():
                parameter.requires_grad_(requires_grad=True)
        elif phase == "flow":
            trainable_modules = [
                self.paligemma_with_expert.gemma_expert,
                self.action_future_in_proj,
                self.action_future_out_proj,
            ]
            if self.pi05:
                trainable_modules.extend([self.time_mlp_in, self.time_mlp_out])
            else:
                trainable_modules.extend(
                    [
                        self.state_proj,
                        self.action_future_time_mlp_in,
                        self.action_future_time_mlp_out,
                    ]
                )
            for module in trainable_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(requires_grad=True)

        # The autoencoder is independent of the policy vision tower. Keep the
        # latter fixed so policy image features remain anchored to the base
        # checkpoint.
        for parameter in self.paligemma_with_expert.paligemma.vision_tower.parameters():
            parameter.requires_grad_(requires_grad=False)
        if phase == "joint":
            for encoder_module in (
                self.future_visuotactile_autoencoder.encoder_in,
                self.future_visuotactile_autoencoder.encoder_stages,
                self.future_visuotactile_autoencoder.encoder_out,
            ):
                for parameter in encoder_module.parameters():
                    parameter.requires_grad_(requires_grad=False)

    def _weighted_reconstruction_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        terms = _reconstruction_loss_terms(pred, target)
        terms["temporal"] = _temporal_difference_loss(pred, target)
        total = (
            self.future_mse_loss_weight * terms["mse"]
            + self.future_charbonnier_loss_weight * terms["charbonnier"]
            + self.future_ssim_loss_weight * terms["ssim"]
            + self.future_gradient_loss_weight * terms["gradient"]
            + self.future_pyramid_loss_weight * terms["pyramid"]
            + self.future_temporal_loss_weight * terms["temporal"]
        )
        return total, terms

    def _autoencoder_forward(self, future_visuotactile: Tensor, *, return_dict: bool):
        future_latents = self.future_visuotactile_autoencoder.encode(future_visuotactile)
        reconstruction = self.future_visuotactile_autoencoder.decode(future_latents)
        autoencoder_loss, terms = self._weighted_reconstruction_loss(reconstruction, future_visuotactile)
        loss = self.future_autoencoder_loss_weight * autoencoder_loss
        if not return_dict:
            return loss
        return {
            "loss": loss,
            "future_autoencoder_loss": autoencoder_loss,
            "future_recon_loss": autoencoder_loss,
            "future_recon_mse": terms["mse"],
            "future_recon_charbonnier": terms["charbonnier"],
            "future_recon_ssim": terms["ssim"],
            "future_recon_gradient": terms["gradient"],
            "future_recon_pyramid": terms["pyramid"],
            "future_recon_temporal": terms["temporal"],
            "pred_future_visuotactile": reconstruction,
            "future_visuotactile_latents": future_latents,
        }

    def _get_future_visuotactile(self, observation, future_visuotactile=None):
        if future_visuotactile is not None:
            return future_visuotactile
        direct = _field(observation, self.future_visuotactile_key)
        if direct is not None:
            return direct
        return _field(observation, "future_tactile")

    def _iter_current_visuotactile(self, observation):
        images = _field(observation, "images", {}) or {}
        image_masks = _field(observation, "image_masks", {}) or {}

        for key in self.visuotactile_keys:
            if key in images:
                yield images[key], image_masks.get(key)

        for field_name, mask_name in (
            ("visuotactile", "visuotactile_mask"),
            ("tactile", "tactile_mask"),
            ("current_visuotactile", "current_visuotactile_mask"),
        ):
            value = _field(observation, field_name)
            if value is None:
                continue
            mask = _field(observation, mask_name)
            if isinstance(value, dict):
                masks = mask if isinstance(mask, dict) else {}
                for key, tensor in value.items():
                    yield tensor, masks.get(key)
            else:
                yield value, mask

    def _to_rgb_image_batch(self, tensor: Tensor) -> tuple[Tensor, int] | None:
        """Return image batch and temporal factor if tensor is image-like."""
        if tensor.ndim == 4:
            if tensor.shape[1] in (1, 3):
                image = tensor
            elif tensor.shape[-1] in (1, 3):
                image = tensor.permute(0, 3, 1, 2)
            else:
                return None
            if image.shape[1] == 1:
                image = image.expand(-1, 3, -1, -1)
            return image, 1

        if tensor.ndim == 5:
            batch_size, steps = tensor.shape[:2]
            if tensor.shape[2] in (1, 3):
                image = tensor.reshape(batch_size * steps, *tensor.shape[2:])
            elif tensor.shape[-1] in (1, 3):
                image = tensor.permute(0, 1, 4, 2, 3).reshape(batch_size * steps, tensor.shape[-1], *tensor.shape[2:4])
            else:
                return None
            if image.shape[1] == 1:
                image = image.expand(-1, 3, -1, -1)
            return image, steps

        return None

    def _embed_visuotactile_tokens(self, tensor: Tensor) -> Tensor:
        image_batch = self._to_rgb_image_batch(tensor)
        if image_batch is not None:
            images, steps = image_batch
            if images.shape[-2:] != (224, 224):
                images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, images)
            if steps > 1:
                batch_size = tensor.shape[0]
                img_emb = img_emb.reshape(batch_size, steps * img_emb.shape[1], img_emb.shape[2])
            return img_emb

        raise ValueError(
            "Current visuo-tactile inputs must be image-like so they can use the PaliGemma vision encoder. "
            f"Got shape {tuple(tensor.shape)}."
        )

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, observation=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed image, current visuo-tactile, and language tokens in the VLM expert."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            batch_size, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks += [0] * num_img_embs

        if observation is not None:
            for vt_tensor, vt_mask in self._iter_current_visuotactile(observation):
                vt_emb = self._embed_visuotactile_tokens(vt_tensor)
                batch_size, num_vt_embs = vt_emb.shape[:2]
                if vt_mask is None:
                    vt_pad_mask = torch.ones(batch_size, num_vt_embs, dtype=torch.bool, device=vt_emb.device)
                elif vt_mask.ndim == 1:
                    vt_pad_mask = vt_mask[:, None].expand(batch_size, num_vt_embs)
                else:
                    repeats = math.ceil(num_vt_embs / vt_mask.shape[1])
                    vt_pad_mask = vt_mask.repeat_interleave(repeats, dim=1)[:, :num_vt_embs]
                embs.append(vt_emb)
                pad_masks.append(vt_pad_mask.to(device=vt_emb.device, dtype=torch.bool))
                att_masks += [0] * num_vt_embs

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_action_future, timestep):
        """Embed state and noisy [action, future_visuotactile_latent] tokens."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)
            state_emb = self._apply_checkpoint(self.state_proj, state)

            embs.append(state_emb[:, None, :])
            batch_size = state_emb.shape[0]
            device = state_emb.device
            pad_masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
            att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_future_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        action_future_emb = self._apply_checkpoint(self.action_future_in_proj, noisy_action_future)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_future_emb)
            action_future_time_emb = torch.cat([action_future_emb, time_emb], dim=2)

            def mlp_func(x):
                x = self.action_future_time_mlp_in(x)
                x = F.silu(x)
                return self.action_future_time_mlp_out(x)

            action_future_time_emb = self._apply_checkpoint(mlp_func, action_future_time_emb)
            adarms_cond = None
        else:

            def time_mlp_func(x):
                x = self.time_mlp_in(x)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)
            action_future_time_emb = action_future_emb

        embs.append(action_future_time_emb)
        batch_size, horizon = action_future_time_emb.shape[:2]
        pad_masks.append(torch.ones(batch_size, horizon, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(batch_size, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond

    def _predict_vector_field(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_att_masks,
        adarms_cond,
    ):
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix, suffix, attention_mask, positions, cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self._apply_checkpoint(self.action_future_out_proj, suffix_out)

    def forward(
        self,
        observation,
        actions,
        future_visuotactile=None,
        noise=None,
        future_noise=None,
        time=None,
        *,
        return_dict: bool = False,
        training_phase: str = "joint",
    ):
        future_visuotactile = self._get_future_visuotactile(observation, future_visuotactile)
        if future_visuotactile is None:
            raise ValueError("future_visuotactile is required for PI0VisuoTactilePytorch.forward.")
        if training_phase == "autoencoder":
            return self._autoencoder_forward(future_visuotactile, return_dict=return_dict)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        future_latents = self.future_visuotactile_autoencoder.encode(future_visuotactile)
        if future_latents.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "future_visuotactile encoder must produce [batch, action_horizon, latent_dim], "
                f"got {tuple(future_latents.shape)} for actions {tuple(actions.shape)}."
            )

        # Flow matching treats encoded data as a fixed target. Without this
        # detach, the encoder can reduce flow loss by mapping every image to a
        # nearly constant latent.
        flow_future_latents = future_latents.detach()
        clean = torch.cat([actions, flow_future_latents], dim=-1)
        action_noise = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        if future_noise is None:
            future_noise = self.sample_noise(flow_future_latents.shape, flow_future_latents.device)
        joint_noise = torch.cat([action_noise, future_noise], dim=-1)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * joint_noise + (1 - time_expanded) * clean
        u_t = joint_noise - clean

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, observation=observation
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        v_t = self._predict_vector_field(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond,
        )

        action_v_t, future_v_t = torch.split(v_t, [self.config.action_dim, self.future_visuotactile_latent_dim], dim=-1)
        action_u_t, future_u_t = torch.split(u_t, [self.config.action_dim, self.future_visuotactile_latent_dim], dim=-1)
        _, noisy_future_latents = torch.split(
            x_t, [self.config.action_dim, self.future_visuotactile_latent_dim], dim=-1
        )

        action_loss = F.mse_loss(action_v_t, action_u_t, reduction="none").mean(dim=-1)
        future_flow_loss = F.mse_loss(future_v_t, future_u_t, reduction="none").mean(dim=-1)
        pred_clean_future_latents = noisy_future_latents - time_expanded * future_v_t
        pred_future_visuotactile = self.future_visuotactile_autoencoder.decode(pred_clean_future_latents)
        future_recon_loss, future_recon_terms = self._weighted_reconstruction_loss(
            pred_future_visuotactile,
            future_visuotactile,
        )

        if training_phase == "joint":
            autoencoder_reconstruction = self.future_visuotactile_autoencoder.decode(future_latents)
            future_autoencoder_loss, _ = self._weighted_reconstruction_loss(
                autoencoder_reconstruction,
                future_visuotactile,
            )
        else:
            future_autoencoder_loss = torch.zeros_like(future_recon_loss)

        loss = (
            self.action_loss_weight * action_loss
            + self.future_flow_loss_weight * future_flow_loss
            + self.future_visuotactile_loss_weight * future_recon_loss
            + self.future_autoencoder_loss_weight * future_autoencoder_loss
        )
        if not return_dict:
            return loss

        return {
            "loss": loss,
            "action_loss": action_loss,
            "future_flow_loss": future_flow_loss,
            "future_recon_loss": future_recon_loss,
            "future_autoencoder_loss": future_autoencoder_loss,
            "future_recon_mse": future_recon_terms["mse"],
            "future_recon_charbonnier": future_recon_terms["charbonnier"],
            "future_recon_ssim": future_recon_terms["ssim"],
            "future_recon_gradient": future_recon_terms["gradient"],
            "future_recon_pyramid": future_recon_terms["pyramid"],
            "future_recon_temporal": future_recon_terms["temporal"],
            "pred_action_velocity": action_v_t,
            "pred_future_velocity": future_v_t,
            "pred_future_visuotactile": pred_future_visuotactile,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        future_noise=None,
        num_steps=10,
        *,
        return_future_visuotactile: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((batch_size, self.config.action_horizon, self.config.action_dim), device)
        if future_noise is None:
            future_noise = self.sample_noise(
                (batch_size, self.config.action_horizon, self.future_visuotactile_latent_dim), device
            )
        x_t = torch.cat([noise, future_noise], dim=-1)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, observation=observation
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt

        actions, future_latents = torch.split(
            x_t, [self.config.action_dim, self.future_visuotactile_latent_dim], dim=-1
        )
        if not return_future_visuotactile:
            return actions
        return {
            "actions": actions,
            "future_visuotactile": self.future_visuotactile_autoencoder.decode(future_latents),
            "future_visuotactile_latents": future_latents,
        }

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        return self.action_future_out_proj(suffix_out)
