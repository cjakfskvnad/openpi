"""PyTorch pi0 with an independent expert for future visuo-tactile prediction.

Unlike :mod:`pi0_visuotactile_pytorch`, this variant does not concatenate
future visuo-tactile latents with actions before the action expert. Actions and
future visuo-tactile latents are embedded by separate Gemma experts and share
attention with the PaliGemma prefix at every transformer layer.
"""

from collections.abc import Iterable
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.models.siglip import check as _siglip_check

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithActionAndVisuoTactileExpertsModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_visuotactile_pytorch import _reconstruction_loss_terms
from openpi.models_pytorch.pi0_visuotactile_pytorch import _temporal_difference_loss
from openpi.models_pytorch.pi0_visuotactile_pytorch import create_future_visuotactile_autoencoder


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


class PI0ExpertVisuoTactilePytorch(PI0Pytorch):
    """pi0 with separate action and future visuo-tactile Gemma experts.

    A standalone convolutional AE defines the target latent space. The tactile
    Gemma stream predicts flow in that space independently of the action
    expert, while all three transformer streams share attention.
    """

    def __init__(self, config):
        # PI0Pytorch.__init__ creates a two-stream model, which can be very
        # large. Initialize nn.Module directly so a redundant model is never
        # allocated before constructing the three-stream version.
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        visuotactile_expert_variant = _config_get(
            config,
            "visuotactile_expert_variant",
            _config_get(config, "visual_tactile_expert_variant", config.action_expert_variant),
        )
        visuotactile_expert_config = _gemma.get_config(visuotactile_expert_variant)

        use_adarms = [False, True, True] if self.pi05 else [False, False, False]
        self.paligemma_with_expert = PaliGemmaWithActionAndVisuoTactileExpertsModel(
            paligemma_config,
            action_expert_config,
            visuotactile_expert_config,
            use_adarms=use_adarms,
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.visuotactile_keys = _as_tuple(
            _config_get(config, "visuotactile_keys", ("visuotactile_0_rgb", "tactile_0_rgb"))
        )
        self.future_visuotactile_key = _config_get(config, "future_visuotactile_key", "future_visuotactile")
        self.action_loss_weight = _config_get(config, "action_loss_weight", 1.0)
        self.future_flow_loss_weight = _config_get(config, "future_flow_loss_weight", 1.0)
        self.future_visuotactile_loss_weight = _config_get(config, "future_visuotactile_loss_weight", 1.0)
        self.future_autoencoder_loss_weight = _config_get(config, "future_autoencoder_loss_weight", 1.0)
        self.future_joint_flow_loss_weight = _config_get(config, "future_joint_flow_loss_weight", 0.05)
        self.future_joint_visuotactile_loss_weight = _config_get(
            config,
            "future_joint_visuotactile_loss_weight",
            0.1,
        )
        self.future_joint_autoencoder_loss_weight = _config_get(
            config,
            "future_joint_autoencoder_loss_weight",
            0.05,
        )
        self.future_mse_loss_weight = _config_get(config, "future_mse_loss_weight", 0.1)
        self.future_charbonnier_loss_weight = _config_get(config, "future_charbonnier_loss_weight", 1.0)
        self.future_ssim_loss_weight = _config_get(config, "future_ssim_loss_weight", 0.2)
        self.future_gradient_loss_weight = _config_get(config, "future_gradient_loss_weight", 0.1)
        self.future_pyramid_loss_weight = _config_get(config, "future_pyramid_loss_weight", 0.1)
        self.future_temporal_loss_weight = _config_get(config, "future_temporal_loss_weight", 0.2)

        self.future_visuotactile_autoencoder = create_future_visuotactile_autoencoder(config)
        self.pretrained_autoencoder_loaded = False
        self.future_visuotactile_latent_dim = self.future_visuotactile_autoencoder.flat_latent_dim

        self.visuotactile_in_proj = nn.Linear(
            self.future_visuotactile_latent_dim,
            visuotactile_expert_config.width,
        )
        self.visuotactile_out_proj = nn.Linear(
            visuotactile_expert_config.width,
            self.future_visuotactile_latent_dim,
        )
        if self.pi05:
            self.visuotactile_time_mlp_in = nn.Linear(
                visuotactile_expert_config.width,
                visuotactile_expert_config.width,
            )
            self.visuotactile_time_mlp_out = nn.Linear(
                visuotactile_expert_config.width,
                visuotactile_expert_config.width,
            )
        else:
            self.visuotactile_time_mlp_in = nn.Linear(
                2 * visuotactile_expert_config.width,
                visuotactile_expert_config.width,
            )
            self.visuotactile_time_mlp_out = nn.Linear(
                visuotactile_expert_config.width,
                visuotactile_expert_config.width,
            )

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)
        self.gradient_checkpointing_enabled = False

        message = (
            "transformers_replace is not installed correctly. Please install it with "
            "`uv pip install transformers==4.53.2` and `cp -r "
            "./src/openpi/models_pytorch/transformers_replace/* "
            ".venv/lib/python3.11/site-packages/transformers/`."
        )
        if not _siglip_check.check_whether_transformers_replace_is_installed_correctly():
            raise ValueError(message)

    def initialize_tactile_expert_from_action_expert(self) -> None:
        """Give the new tactile expert the pretrained action expert initialization."""
        self.paligemma_with_expert.gemma_visuotactile_expert.load_state_dict(
            self.paligemma_with_expert.gemma_expert.state_dict(),
            strict=True,
        )

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_visuotactile_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_visuotactile_expert.model.gradient_checkpointing = False

    def get_training_phase(self, step: int) -> str:
        """Train the tactile latent expert first, then jointly fine-tune."""
        if not self.pretrained_autoencoder_loaded:
            raise RuntimeError(
                "PI0ExpertVisuoTactilePytorch requires a pretrained autoencoder. "
                "Set --pytorch-autoencoder-weight-path to a standalone AE checkpoint."
            )
        joint_start_step = _config_get(self.config, "future_joint_finetune_start_step", 5_000)
        return "latent" if step < joint_start_step else "joint"

    def set_train_phase(self, phase: str) -> None:
        """Select exactly the parameters intended for each training stage."""
        if phase not in {"latent", "joint"}:
            raise ValueError(f"Unknown expert visuo-tactile training phase: {phase}.")

        for parameter in self.parameters():
            parameter.requires_grad_(phase == "joint")

        if phase == "latent":
            tactile_modules = [
                self.paligemma_with_expert.gemma_visuotactile_expert,
                self.visuotactile_in_proj,
                self.visuotactile_out_proj,
                self.visuotactile_time_mlp_in,
                self.visuotactile_time_mlp_out,
            ]
            for module in tactile_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(requires_grad=True)

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

    def _future_loss_weights(self, training_phase: str) -> tuple[float, float, float]:
        if training_phase == "latent":
            return (
                self.future_flow_loss_weight,
                self.future_visuotactile_loss_weight,
                0.0,
            )
        return (
            self.future_joint_flow_loss_weight,
            self.future_joint_visuotactile_loss_weight,
            self.future_joint_autoencoder_loss_weight,
        )

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

    @staticmethod
    def _to_rgb_image_batch(tensor: Tensor) -> tuple[Tensor, int] | None:
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
                image = tensor.permute(0, 1, 4, 2, 3).reshape(
                    batch_size * steps,
                    tensor.shape[-1],
                    *tensor.shape[2:4],
                )
            else:
                return None
            if image.shape[1] == 1:
                image = image.expand(-1, 3, -1, -1)
            return image, steps
        return None

    def _embed_visuotactile_tokens(self, tensor: Tensor) -> Tensor:
        image_batch = self._to_rgb_image_batch(tensor)
        if image_batch is None:
            raise ValueError(
                "Current visuo-tactile inputs must be image-like so they can use "
                f"the PaliGemma vision encoder. Got shape {tuple(tensor.shape)}."
            )
        images, steps = image_batch
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        image_embeddings = self._apply_checkpoint(self.paligemma_with_expert.embed_image, images)
        if steps > 1:
            batch_size = tensor.shape[0]
            image_embeddings = image_embeddings.reshape(
                batch_size,
                steps * image_embeddings.shape[1],
                image_embeddings.shape[2],
            )
        return image_embeddings

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        observation=None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = []
        pad_masks = []
        attention_ar = []

        for image, image_mask in zip(images, img_masks, strict=True):
            image_embedding = self._apply_checkpoint(self.paligemma_with_expert.embed_image, image)
            batch_size, token_count = image_embedding.shape[:2]
            embeddings.append(image_embedding)
            pad_masks.append(image_mask[:, None].expand(batch_size, token_count))
            attention_ar += [0] * token_count

        if observation is not None:
            for tactile, tactile_mask in self._iter_current_visuotactile(observation):
                tactile_embedding = self._embed_visuotactile_tokens(tactile)
                batch_size, token_count = tactile_embedding.shape[:2]
                if tactile_mask is None:
                    token_mask = torch.ones(
                        batch_size,
                        token_count,
                        dtype=torch.bool,
                        device=tactile_embedding.device,
                    )
                elif tactile_mask.ndim == 1:
                    token_mask = tactile_mask[:, None].expand(batch_size, token_count)
                else:
                    repeats = math.ceil(token_count / tactile_mask.shape[1])
                    token_mask = tactile_mask.repeat_interleave(repeats, dim=1)[:, :token_count]
                embeddings.append(tactile_embedding)
                pad_masks.append(token_mask.to(device=tactile_embedding.device, dtype=torch.bool))
                attention_ar += [0] * token_count

        def embed_language(tokens):
            embedding = self.paligemma_with_expert.embed_language_tokens(tokens)
            return embedding * math.sqrt(embedding.shape[-1])

        language_embedding = self._apply_checkpoint(embed_language, lang_tokens)
        embeddings.append(language_embedding)
        pad_masks.append(lang_masks)
        attention_ar += [0] * language_embedding.shape[1]

        embeddings = torch.cat(embeddings, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        attention_ar = torch.tensor(attention_ar, dtype=torch.bool, device=pad_masks.device)
        attention_ar = attention_ar[None, :].expand(pad_masks.shape[0], -1)
        return embeddings, pad_masks, attention_ar

    def embed_action_suffix(self, state, noisy_actions, timestep):
        return PI0Pytorch.embed_suffix(self, state, noisy_actions, timestep)

    def embed_visuotactile_suffix(self, noisy_latents: Tensor, timestep: Tensor):
        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.visuotactile_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        ).to(dtype=timestep.dtype)
        latent_embedding = self._apply_checkpoint(self.visuotactile_in_proj, noisy_latents)

        if self.pi05:

            def time_mlp(value):
                value = self.visuotactile_time_mlp_in(value)
                value = F.silu(value)
                value = self.visuotactile_time_mlp_out(value)
                return F.silu(value)

            adarms_cond = self._apply_checkpoint(time_mlp, time_embedding)
            expert_embedding = latent_embedding
        else:
            time_embedding = time_embedding[:, None, :].expand_as(latent_embedding)

            def fuse_time(value):
                value = self.visuotactile_time_mlp_in(value)
                value = F.silu(value)
                return self.visuotactile_time_mlp_out(value)

            expert_embedding = self._apply_checkpoint(
                fuse_time,
                torch.cat([latent_embedding, time_embedding], dim=-1),
            )
            adarms_cond = None

        batch_size, horizon = expert_embedding.shape[:2]
        pad_mask = torch.ones(batch_size, horizon, dtype=torch.bool, device=expert_embedding.device)
        # All action and tactile suffix tokens belong to one bidirectional
        # attention block. The action stream introduces the block boundary.
        attention_ar = torch.zeros(batch_size, horizon, dtype=expert_embedding.dtype, device=expert_embedding.device)
        return expert_embedding, pad_mask, attention_ar, adarms_cond

    def _predict_vector_fields(
        self,
        prefix,
        prefix_pad_mask,
        prefix_attention_ar,
        action_suffix,
        action_pad_mask,
        action_attention_ar,
        visuotactile_suffix,
        visuotactile_pad_mask,
        visuotactile_attention_ar,
        action_adarms_cond,
        visuotactile_adarms_cond,
    ) -> tuple[Tensor, Tensor]:
        target_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix = prefix.to(dtype=target_dtype)
        action_suffix = action_suffix.to(dtype=target_dtype)
        visuotactile_suffix = visuotactile_suffix.to(dtype=target_dtype)

        pad_mask = torch.cat([prefix_pad_mask, action_pad_mask, visuotactile_pad_mask], dim=1)
        attention_ar = torch.cat(
            [prefix_attention_ar, action_attention_ar, visuotactile_attention_ar],
            dim=1,
        )
        attention_mask = self._prepare_attention_masks_4d(make_att_2d_masks(pad_mask, attention_ar))
        position_ids = torch.cumsum(pad_mask, dim=1) - 1

        def joint_forward(prefix_tokens, action_tokens, tactile_tokens, mask, positions, action_cond, tactile_cond):
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix_tokens, action_tokens, tactile_tokens],
                use_cache=False,
                adarms_cond=[None, action_cond, tactile_cond],
            )
            return outputs[1], outputs[2]

        action_output, visuotactile_output = self._apply_checkpoint(
            joint_forward,
            prefix,
            action_suffix,
            visuotactile_suffix,
            attention_mask,
            position_ids,
            action_adarms_cond,
            visuotactile_adarms_cond,
        )
        action_output = action_output[:, -self.config.action_horizon :].to(dtype=torch.float32)
        visuotactile_output = visuotactile_output[:, -self.config.action_horizon :].to(dtype=torch.float32)
        action_velocity = self._apply_checkpoint(self.action_out_proj, action_output)
        visuotactile_velocity = self._apply_checkpoint(self.visuotactile_out_proj, visuotactile_output)
        return action_velocity, visuotactile_velocity

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
        if training_phase not in {"latent", "joint"}:
            raise ValueError(f"Unknown expert visuo-tactile training phase: {training_phase}.")
        images, image_masks, language_tokens, language_masks, state = self._preprocess_observation(
            observation,
            train=True,
        )
        future_visuotactile = self._get_future_visuotactile(observation, future_visuotactile)
        if future_visuotactile is None:
            raise ValueError("future_visuotactile is required for PI0ExpertVisuoTactilePytorch.forward.")

        encoded_future_latents = self.future_visuotactile_autoencoder.encode(future_visuotactile)
        if encoded_future_latents.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "The future autoencoder must produce [batch, action_horizon, latent_dim], "
                f"got {tuple(encoded_future_latents.shape)} for actions {tuple(actions.shape)}."
            )
        # The codec defines the target latent coordinate system. Flow matching
        # never updates its encoder through the target branch, including during
        # joint fine-tuning; joint AE updates come from reconstruction instead.
        future_latents = encoded_future_latents.detach()

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if future_noise is None:
            future_noise = self.sample_noise(future_latents.shape, future_latents.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        noisy_future_latents = time_expanded * future_noise + (1 - time_expanded) * future_latents
        action_target_velocity = noise - actions
        visuotactile_target_velocity = future_noise - future_latents

        prefix = self.embed_prefix(
            images,
            image_masks,
            language_tokens,
            language_masks,
            observation=observation,
        )
        action_suffix = self.embed_action_suffix(state, noisy_actions, time)
        visuotactile_suffix = self.embed_visuotactile_suffix(noisy_future_latents, time)
        action_velocity, visuotactile_velocity = self._predict_vector_fields(
            *prefix,
            *action_suffix[:3],
            *visuotactile_suffix[:3],
            action_suffix[3],
            visuotactile_suffix[3],
        )

        action_loss = F.mse_loss(action_velocity, action_target_velocity, reduction="none").mean(dim=-1)
        future_flow_loss = F.mse_loss(
            visuotactile_velocity,
            visuotactile_target_velocity,
            reduction="none",
        ).mean(dim=-1)

        predicted_clean_latents = noisy_future_latents - time_expanded * visuotactile_velocity
        predicted_future_visuotactile = self.future_visuotactile_autoencoder.decode(predicted_clean_latents)
        future_reconstruction_loss, future_reconstruction_terms = self._weighted_reconstruction_loss(
            predicted_future_visuotactile,
            future_visuotactile,
        )

        if training_phase == "latent":
            # The pretrained AE is frozen in this phase, so its self-
            # reconstruction loss cannot update any trainable parameter. Skip
            # the redundant decode and report a zero term.
            future_autoencoder_loss = torch.zeros_like(future_reconstruction_loss)
        else:
            autoencoder_reconstruction = self.future_visuotactile_autoencoder.decode(encoded_future_latents)
            future_autoencoder_loss, _ = self._weighted_reconstruction_loss(
                autoencoder_reconstruction,
                future_visuotactile,
            )

        flow_loss_weight, reconstruction_loss_weight, autoencoder_loss_weight = self._future_loss_weights(
            training_phase
        )
        future_loss = (
            flow_loss_weight * future_flow_loss
            + reconstruction_loss_weight * future_reconstruction_loss
            + autoencoder_loss_weight * future_autoencoder_loss
        )
        # Keep action supervision at full strength in both phases. During the
        # first phase set_train_phase() restricts gradients to the tactile
        # prediction expert; joint training unfreezes the full model and
        # downweights the auxiliary future objectives.
        loss = self.action_loss_weight * action_loss + future_loss

        if not return_dict:
            return loss
        return {
            "loss": loss,
            "action_loss": action_loss,
            "future_loss": future_loss,
            "future_flow_loss": future_flow_loss,
            "future_recon_loss": future_reconstruction_loss,
            "future_autoencoder_loss": future_autoencoder_loss,
            "future_recon_mse": future_reconstruction_terms["mse"],
            "future_recon_charbonnier": future_reconstruction_terms["charbonnier"],
            "future_recon_ssim": future_reconstruction_terms["ssim"],
            "future_recon_gradient": future_reconstruction_terms["gradient"],
            "future_recon_pyramid": future_reconstruction_terms["pyramid"],
            "future_recon_temporal": future_reconstruction_terms["temporal"],
            "pred_action_velocity": action_velocity,
            "pred_future_velocity": visuotactile_velocity,
            "pred_future_visuotactile": predicted_future_visuotactile,
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
            noise = self.sample_noise(
                (batch_size, self.config.action_horizon, self.config.action_dim),
                device,
            )
        if future_noise is None:
            future_noise = self.sample_noise(
                (
                    batch_size,
                    self.config.action_horizon,
                    self.future_visuotactile_latent_dim,
                ),
                device,
            )

        images, image_masks, language_tokens, language_masks, state = self._preprocess_observation(
            observation,
            train=False,
        )
        prefix = self.embed_prefix(
            images,
            image_masks,
            language_tokens,
            language_masks,
            observation=observation,
        )

        actions = noise
        future_latents = future_noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            action_velocity, visuotactile_velocity = self.denoise_step(
                state,
                prefix,
                actions,
                future_latents,
                expanded_time,
            )
            actions = actions + dt * action_velocity
            future_latents = future_latents + dt * visuotactile_velocity
            time += dt

        if not return_future_visuotactile:
            return actions
        return {
            "actions": actions,
            "future_visuotactile": self.future_visuotactile_autoencoder.decode(future_latents),
            "future_visuotactile_latents": future_latents,
        }

    def denoise_step(
        self,
        state,
        prefix,
        noisy_actions,
        noisy_future_latents,
        timestep,
    ) -> tuple[Tensor, Tensor]:
        action_suffix = self.embed_action_suffix(state, noisy_actions, timestep)
        visuotactile_suffix = self.embed_visuotactile_suffix(noisy_future_latents, timestep)
        return self._predict_vector_fields(
            *prefix,
            *action_suffix[:3],
            *visuotactile_suffix[:3],
            action_suffix[3],
            visuotactile_suffix[3],
        )
