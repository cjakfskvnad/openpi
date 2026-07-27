"""Standalone future visuo-tactile autoencoder and reconstruction losses."""

import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812


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
    """Self-contained convolutional autoencoder for future images."""

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
    losses = []
    for _ in range(levels):
        losses.append((pred - target).abs().flatten(1).mean(1))
        if min(pred.shape[-2:]) <= 8:
            break
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target = F.avg_pool2d(target, kernel_size=2, stride=2)
    return torch.stack(losses).mean(0)


def reconstruction_loss_terms(pred: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Return per-[batch, horizon] image reconstruction terms."""
    batch_size, horizon = pred.shape[:2]
    pred_nchw = _future_images_to_nchw(pred).to(dtype=torch.float32)
    target_nchw = _future_images_to_nchw(target).to(dtype=torch.float32)
    difference = pred_nchw - target_nchw

    def reshape(value: Tensor) -> Tensor:
        return value.reshape(batch_size, horizon)

    return {
        "mse": reshape(difference.square().flatten(1).mean(1)),
        "charbonnier": reshape(torch.sqrt(difference.square() + 1e-6).flatten(1).mean(1)),
        "ssim": reshape(_ssim_loss(pred_nchw, target_nchw)),
        "gradient": reshape(_gradient_loss(pred_nchw, target_nchw)),
        "pyramid": reshape(_pyramid_loss(pred_nchw, target_nchw)),
    }


def temporal_difference_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Match frame-to-frame motion and return a [batch, horizon] loss."""
    pred_nchw = _future_images_to_nchw(pred)
    target_nchw = _future_images_to_nchw(target)
    batch_size, horizon = pred.shape[:2]
    pred_nchw = pred_nchw.reshape(batch_size, horizon, *pred_nchw.shape[1:])
    target_nchw = target_nchw.reshape(batch_size, horizon, *target_nchw.shape[1:])
    temporal = (pred_nchw[:, 1:] - pred_nchw[:, :-1]) - (target_nchw[:, 1:] - target_nchw[:, :-1])
    temporal = temporal.to(dtype=torch.float32).abs().flatten(2).mean(2)
    return F.pad(temporal, (1, 0))


def create_future_visuotactile_autoencoder(config: Any) -> FutureVisuoTactileVisionAutoencoder:
    """Build the standalone future-image codec from a Pi0 visuo-tactile config."""
    future_shape = getattr(config, "future_visuotactile_shape", None)
    if future_shape is None:
        raise ValueError("future_visuotactile_shape is required to create the autoencoder.")
    return FutureVisuoTactileVisionAutoencoder(
        latent_dim=getattr(config, "future_visuotactile_latent_dim", 16),
        future_shape=tuple(future_shape),
        encoder_width=getattr(config, "future_visuotactile_encoder_width", 64),
        decoder_width=getattr(config, "future_visuotactile_decoder_width", 256),
        decoder_depth=getattr(config, "future_visuotactile_decoder_depth", 2),
        latent_grid_size=getattr(config, "future_visuotactile_latent_grid_size", None),
    )


__all__ = [
    "FutureVisuoTactileVisionAutoencoder",
    "create_future_visuotactile_autoencoder",
    "reconstruction_loss_terms",
    "temporal_difference_loss",
]
