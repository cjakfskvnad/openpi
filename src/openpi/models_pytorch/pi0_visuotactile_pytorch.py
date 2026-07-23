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


class FutureVisuoTactileVisionAutoencoder(nn.Module):
    """Vision-style future visuo-tactile encoder and patch decoder.

    The encoder reuses the PaliGemma/SigLIP image encoder. Each future tactile
    frame is encoded as image tokens, mean-pooled, then projected to one latent
    token per action timestep. The decoder is a ViT-style patch decoder with
    learned patch queries conditioned on the denoised future latent.
    """

    def __init__(
        self,
        *,
        image_encoder,
        encoder_width: int,
        latent_dim: int,
        future_shape: tuple[int, ...],
        decoder_width: int,
        decoder_depth: int,
        decoder_num_heads: int,
        patch_size: int,
        encoder_chunk_size: int,
        encoder_image_size: int = 224,
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

        self.image_encoder = image_encoder
        self.latent_dim = latent_dim
        self.future_shape = future_shape
        self.out_channels = channels
        self.out_height = height
        self.out_width = width
        self.patch_size = patch_size
        self.encoder_chunk_size = encoder_chunk_size
        self.encoder_image_size = encoder_image_size
        self.gradient_checkpointing_enabled = False

        self.encoder_proj = nn.Linear(encoder_width, latent_dim)
        self.latent_to_decoder = nn.Linear(latent_dim, decoder_width)

        self.patch_grid_h = math.ceil(height / patch_size)
        self.patch_grid_w = math.ceil(width / patch_size)
        self.num_patches = self.patch_grid_h * self.patch_grid_w
        self.patch_queries = nn.Parameter(torch.zeros(1, self.num_patches, decoder_width))
        self.patch_pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, decoder_width))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_width,
            nhead=decoder_num_heads,
            dim_feedforward=4 * decoder_width,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.patch_out = nn.Linear(decoder_width, channels * patch_size * patch_size)
        nn.init.normal_(self.patch_queries, std=0.02)
        nn.init.normal_(self.patch_pos_embedding, std=0.02)

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
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        return images, batch_size, horizon

    def encode(self, future_visuotactile: Tensor) -> Tensor:
        images, batch_size, horizon = self._future_to_image_batch(future_visuotactile)
        if images.shape[-2:] != (self.encoder_image_size, self.encoder_image_size):
            images = F.interpolate(
                images, size=(self.encoder_image_size, self.encoder_image_size), mode="bilinear", align_corners=False
            )

        pooled_chunks = []
        for image_chunk in torch.split(images, self.encoder_chunk_size, dim=0):
            if self.gradient_checkpointing_enabled and self.training:
                image_tokens = torch.utils.checkpoint.checkpoint(
                    self.image_encoder,
                    image_chunk,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                image_tokens = self.image_encoder(image_chunk)
            pooled_chunks.append(image_tokens.mean(dim=1).to(dtype=torch.float32))

        pooled = torch.cat(pooled_chunks, dim=0)
        latents = self.encoder_proj(pooled)
        return latents.reshape(batch_size, horizon, self.latent_dim)

    def decode(self, future_latents: Tensor) -> Tensor:
        batch_size, horizon = future_latents.shape[:2]
        latents = future_latents.reshape(batch_size * horizon, self.latent_dim)
        cond = self.latent_to_decoder(latents)[:, None, :]
        tokens = self.patch_queries + self.patch_pos_embedding + cond
        tokens = self.decoder(tokens)
        patches = self.patch_out(tokens)

        patches = patches.reshape(
            batch_size * horizon,
            self.patch_grid_h,
            self.patch_grid_w,
            self.out_channels,
            self.patch_size,
            self.patch_size,
        )
        image = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size * horizon,
            self.out_channels,
            self.patch_grid_h * self.patch_size,
            self.patch_grid_w * self.patch_size,
        )
        image = image[:, :, : self.out_height, : self.out_width]
        if self.channels_first:
            return image.reshape(batch_size, horizon, self.out_channels, self.out_height, self.out_width)
        return image.reshape(batch_size, horizon, self.out_channels, self.out_height, self.out_width).permute(
            0, 1, 3, 4, 2
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

        vlm_width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        expert_width = self.action_in_proj.out_features

        self.visuotactile_keys = _as_tuple(
            _config_get(config, "visuotactile_keys", ("visuotactile_0_rgb", "tactile_0_rgb"))
        )
        self.future_visuotactile_key = _config_get(config, "future_visuotactile_key", "future_visuotactile")
        self.future_visuotactile_latent_dim = _config_get(
            config, "future_visuotactile_latent_dim", config.action_dim
        )
        self.action_loss_weight = _config_get(config, "action_loss_weight", 1.0)
        self.future_flow_loss_weight = _config_get(config, "future_flow_loss_weight", 1.0)
        self.future_visuotactile_loss_weight = _config_get(config, "future_visuotactile_loss_weight", 1.0)

        future_shape = _config_get(config, "future_visuotactile_shape", None)
        if future_shape is None:
            raise ValueError("PI0VisuoTactilePytorch requires config.future_visuotactile_shape.")
        self.future_visuotactile_autoencoder = FutureVisuoTactileVisionAutoencoder(
            image_encoder=self.paligemma_with_expert.embed_image,
            encoder_width=vlm_width,
            latent_dim=self.future_visuotactile_latent_dim,
            future_shape=tuple(future_shape),
            decoder_width=_config_get(config, "future_visuotactile_decoder_width", 512),
            decoder_depth=_config_get(config, "future_visuotactile_decoder_depth", 4),
            decoder_num_heads=_config_get(config, "future_visuotactile_decoder_num_heads", 8),
            patch_size=_config_get(config, "future_visuotactile_patch_size", 16),
            encoder_chunk_size=_config_get(config, "future_visuotactile_encoder_chunk_size", 8),
        )

        self.joint_action_dim = config.action_dim + self.future_visuotactile_latent_dim
        self.action_future_in_proj = nn.Linear(self.joint_action_dim, expert_width)
        self.action_future_out_proj = nn.Linear(expert_width, self.joint_action_dim)

        if not self.pi05:
            self.action_future_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
            self.action_future_time_mlp_out = nn.Linear(expert_width, expert_width)

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        self.future_visuotactile_autoencoder.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()
        self.future_visuotactile_autoencoder.gradient_checkpointing_enabled = False

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

    def _predict_vector_field(self, prefix_embs, prefix_pad_masks, prefix_att_masks, suffix_embs, suffix_pad_masks,
                              suffix_att_masks, adarms_cond):
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
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
    ):
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        future_visuotactile = self._get_future_visuotactile(observation, future_visuotactile)
        if future_visuotactile is None:
            raise ValueError("future_visuotactile is required for PI0VisuoTactilePytorch.forward.")

        future_latents = self.future_visuotactile_autoencoder.encode(future_visuotactile)
        if future_latents.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "future_visuotactile encoder must produce [batch, action_horizon, latent_dim], "
                f"got {tuple(future_latents.shape)} for actions {tuple(actions.shape)}."
            )

        clean = torch.cat([actions, future_latents], dim=-1)
        action_noise = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        if future_noise is None:
            future_noise = self.sample_noise(future_latents.shape, future_latents.device)
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
            prefix_embs, prefix_pad_masks, prefix_att_masks, suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond
        )

        action_v_t, future_v_t = torch.split(v_t, [self.config.action_dim, self.future_visuotactile_latent_dim], dim=-1)
        action_u_t, future_u_t = torch.split(
            u_t, [self.config.action_dim, self.future_visuotactile_latent_dim], dim=-1
        )
        _, noisy_future_latents = torch.split(
            x_t, [self.config.action_dim, self.future_visuotactile_latent_dim], dim=-1
        )

        action_loss = F.mse_loss(action_v_t, action_u_t, reduction="none").mean(dim=-1)
        future_flow_loss = F.mse_loss(future_v_t, future_u_t, reduction="none").mean(dim=-1)
        pred_clean_future_latents = noisy_future_latents - time_expanded * future_v_t
        pred_future_visuotactile = self.future_visuotactile_autoencoder.decode(pred_clean_future_latents)
        future_recon_loss = F.mse_loss(pred_future_visuotactile, future_visuotactile, reduction="none").flatten(2).mean(2)

        loss = (
            self.action_loss_weight * action_loss
            + self.future_flow_loss_weight * future_flow_loss
            + self.future_visuotactile_loss_weight * future_recon_loss
        )
        if not return_dict:
            return loss

        return {
            "loss": loss,
            "action_loss": action_loss,
            "future_flow_loss": future_flow_loss,
            "future_recon_loss": future_recon_loss,
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
