"""PyTorch pi0 with an independent expert for future visuo-tactile prediction.

Unlike :mod:`pi0_visuotactile_pytorch`, this variant does not concatenate
future visuo-tactile latents with actions before the action expert. Actions and
future visuo-tactile latents are embedded by separate Gemma experts and share
attention with the PaliGemma prefix at every transformer layer.
"""

from collections.abc import Iterable, Mapping
import copy
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import DynamicCache
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


class FourierStateEncoder(nn.Module):
    """Encode each continuous state dimension with Fourier features and an MLP."""

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        fourier_dim: int = 8,
        min_period: float = 1e-3,
        max_period: float = 1.0,
    ):
        super().__init__()
        if fourier_dim % 2:
            raise ValueError("fourier_dim must be divisible by 2.")
        self.fourier_dim = fourier_dim
        periods = torch.logspace(
            math.log10(min_period),
            math.log10(max_period),
            fourier_dim // 2,
        )
        self.register_buffer("frequencies", 2 * math.pi / periods, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim * (fourier_dim + 1), output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        frequencies = self.frequencies.to(device=state.device, dtype=torch.float32)
        state_float = state.to(dtype=torch.float32)
        angles = state_float.unsqueeze(-1) * frequencies
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(start_dim=-2)
        return self.mlp(torch.cat([state_float, fourier], dim=-1))


class IndependentTactileSiglipEncoder(nn.Module):
    """An independent copy of PaliGemma's image feature extraction path."""

    def __init__(self, paligemma_model: nn.Module):
        super().__init__()
        self.vision_tower = copy.deepcopy(paligemma_model.vision_tower)
        self.multi_modal_projector = copy.deepcopy(paligemma_model.multi_modal_projector)
        self.config = copy.deepcopy(paligemma_model.config)
        # Reuse the installed implementation so feature scaling stays exactly
        # aligned with the regular image path.
        self._get_image_features_impl = type(paligemma_model).get_image_features

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self._get_image_features_impl(self, pixel_values)


class PI0ExpertVisuoTactilePytorch(PI0Pytorch):
    """pi0 with separate action and future visuo-tactile Gemma experts.

    A standalone convolutional AE defines the target latent space. The tactile
    Gemma stream predicts flow in that space independently of the action
    expert, while all three transformer streams share attention.
    """

    _TACTILE_ENCODER_STATE_PREFIX = "tactile_encoder."

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

        self.state_input_mode = _config_get(config, "state_input_mode", "none")
        if self.state_input_mode not in {"none", "adarms"}:
            raise ValueError(
                "state_input_mode must be one of {'none', 'adarms'}, "
                f"got {self.state_input_mode!r}."
            )
        if self.state_input_mode == "adarms" and not self.pi05:
            raise ValueError("state_input_mode='adarms' requires pi05=True.")

        use_adarms = [False, True, True] if self.pi05 else [False, False, False]
        self.paligemma_with_expert = PaliGemmaWithActionAndVisuoTactileExpertsModel(
            paligemma_config,
            action_expert_config,
            visuotactile_expert_config,
            use_adarms=use_adarms,
            action_adarms_cond_dim=(
                2 * action_expert_config.width if self.state_input_mode == "adarms" else None
            ),
            precision=config.dtype,
        )
        self.tactile_encoder = IndependentTactileSiglipEncoder(
            self.paligemma_with_expert.paligemma.model
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        if getattr(self, "state_input_mode", "none") == "adarms":
            self.state_encoder = FourierStateEncoder(config.action_dim, action_expert_config.width)
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

    def initialize_tactile_encoder_from_image_encoder(self) -> None:
        """Copy the current image encoder weights into the independent tactile path."""
        paligemma = self.paligemma_with_expert.paligemma
        self.tactile_encoder.vision_tower.load_state_dict(
            paligemma.vision_tower.state_dict(), strict=True
        )
        self.tactile_encoder.multi_modal_projector.load_state_dict(
            paligemma.multi_modal_projector.state_dict(), strict=True
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,  # noqa: FBT001, FBT002
        assign: bool = False,  # noqa: FBT001, FBT002
    ):
        """Load older checkpoints, adapting newly added tactile/state modules."""
        state_dict = dict(state_dict)
        checkpoint_has_tactile_encoder = any(
            key.startswith(self._TACTILE_ENCODER_STATE_PREFIX) for key in state_dict
        )
        if getattr(self, "state_input_mode", "none") == "adarms":
            model_state = self.state_dict()
            dense_suffixes = (
                ".input_layernorm.dense.weight",
                ".post_attention_layernorm.dense.weight",
                ".norm.dense.weight",
            )
            for key, checkpoint_value in tuple(state_dict.items()):
                if not key.startswith("paligemma_with_expert.gemma_expert."):
                    continue
                if not key.endswith(dense_suffixes) or key not in model_state:
                    continue
                model_value = model_state[key]
                if (
                    checkpoint_value.ndim == 2
                    and model_value.ndim == 2
                    and checkpoint_value.shape[0] == model_value.shape[0]
                    and 2 * checkpoint_value.shape[1] == model_value.shape[1]
                ):
                    # The old condition contains timestep only. The new
                    # condition is [state, timestep], so zero-initialize the
                    # state columns and preserve the old timestep projection.
                    expanded = checkpoint_value.new_zeros(model_value.shape)
                    expanded[:, checkpoint_value.shape[1] :] = checkpoint_value
                    state_dict[key] = expanded
        incompatible_keys = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if not checkpoint_has_tactile_encoder:
            self.initialize_tactile_encoder_from_image_encoder()
        return incompatible_keys

    def initialize_tactile_expert_from_action_expert(self) -> None:
        """Give the new tactile expert the pretrained action expert initialization."""
        action_state = self.paligemma_with_expert.gemma_expert.state_dict()
        tactile_state = self.paligemma_with_expert.gemma_visuotactile_expert.state_dict()
        compatible_state = {}
        for key, action_value in action_state.items():
            tactile_value = tactile_state[key]
            if action_value.shape == tactile_value.shape:
                compatible_state[key] = action_value
            elif (
                action_value.ndim == 2
                and tactile_value.ndim == 2
                and action_value.shape[0] == tactile_value.shape[0]
                and action_value.shape[1] == 2 * tactile_value.shape[1]
                and key.endswith(
                    (
                        ".input_layernorm.dense.weight",
                        ".post_attention_layernorm.dense.weight",
                        ".norm.dense.weight",
                    )
                )
            ):
                # The action expert condition is [state, timestep], whereas
                # the tactile expert is conditioned by timestep only.
                compatible_state[key] = action_value[:, tactile_value.shape[1] :]
            else:
                raise ValueError(
                    "Cannot initialize tactile expert parameter "
                    f"{key}: action shape {tuple(action_value.shape)} does not "
                    f"match tactile shape {tuple(tactile_value.shape)}."
                )
        self.paligemma_with_expert.gemma_visuotactile_expert.load_state_dict(
            compatible_state,
            strict=True,
        )

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.tactile_encoder.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_visuotactile_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.tactile_encoder.vision_tower.gradient_checkpointing = False
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
                self.tactile_encoder,
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

    @staticmethod
    def _action_flow_loss(
        action_velocity: Tensor,
        action_target_velocity: Tensor,
        training_phase: str,
    ) -> Tensor:
        """Disable action supervision entirely during tactile-only warmup."""
        if training_phase == "latent":
            return action_velocity.new_zeros(action_velocity.shape[:2])
        return F.mse_loss(
            action_velocity,
            action_target_velocity,
            reduction="none",
        ).mean(dim=-1)

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
        image_embeddings = self._apply_checkpoint(self.tactile_encoder, images)
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
        embeddings, pad_mask, attention_ar, adarms_cond = PI0Pytorch.embed_suffix(
            self, state, noisy_actions, timestep
        )
        if self.state_input_mode == "adarms":
            state_embedding = self._apply_checkpoint(self.state_encoder, state)
            if state_embedding.ndim == 3:
                state_embedding = state_embedding.mean(dim=1)
            state_embedding = F.layer_norm(state_embedding, (state_embedding.shape[-1],))
            adarms_cond = torch.cat([state_embedding, adarms_cond], dim=-1)
        return embeddings, pad_mask, attention_ar, adarms_cond

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
        attention_ar = torch.zeros(
            batch_size, horizon, dtype=expert_embedding.dtype, device=expert_embedding.device
        )
        return expert_embedding, pad_mask, attention_ar, adarms_cond

    @staticmethod
    def _isolate_action_and_tactile_attention(
        attention_mask: Tensor,
        action_start: int,
        tactile_start: int,
    ) -> Tensor:
        """Remove both cross-stream directions while preserving prefix access."""
        attention_mask = attention_mask.clone()
        attention_mask[:, action_start:tactile_start, tactile_start:] = False
        attention_mask[:, tactile_start:, action_start:tactile_start] = False
        return attention_mask

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
        prefix_len = prefix_pad_mask.shape[1]
        action_len = action_pad_mask.shape[1]
        attention_mask = make_att_2d_masks(pad_mask, attention_ar)
        attention_mask = self._isolate_action_and_tactile_attention(
            attention_mask,
            action_start=prefix_len,
            tactile_start=prefix_len + action_len,
        )
        attention_mask = self._prepare_attention_masks_4d(attention_mask)
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

    def _predict_vector_fields_with_prefix_cache(
        self,
        prefix_pad_mask,
        past_key_values,
        action_suffix,
        action_pad_mask,
        action_attention_ar,
        visuotactile_suffix,
        visuotactile_pad_mask,
        visuotactile_attention_ar,
        action_adarms_cond,
        visuotactile_adarms_cond,
    ) -> tuple[Tensor, Tensor]:
        target_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        action_suffix = action_suffix.to(dtype=target_dtype)
        visuotactile_suffix = visuotactile_suffix.to(dtype=target_dtype)

        suffix_pad_mask = torch.cat([action_pad_mask, visuotactile_pad_mask], dim=1)
        suffix_attention_ar = torch.cat(
            [action_attention_ar, visuotactile_attention_ar],
            dim=1,
        )
        batch_size, suffix_len = suffix_pad_mask.shape
        prefix_len = prefix_pad_mask.shape[1]
        prefix_attention_mask = prefix_pad_mask[:, None, :].expand(
            batch_size,
            suffix_len,
            prefix_len,
        )
        suffix_attention_mask = make_att_2d_masks(suffix_pad_mask, suffix_attention_ar)
        suffix_attention_mask = self._isolate_action_and_tactile_attention(
            suffix_attention_mask,
            action_start=0,
            tactile_start=action_pad_mask.shape[1],
        )
        attention_mask = self._prepare_attention_masks_4d(
            torch.cat([prefix_attention_mask, suffix_attention_mask], dim=2)
        )

        prefix_offsets = torch.sum(prefix_pad_mask, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_mask, dim=1) - 1

        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, action_suffix, visuotactile_suffix],
            use_cache=False,
            adarms_cond=[None, action_adarms_cond, visuotactile_adarms_cond],
        )
        action_output = outputs[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        visuotactile_output = outputs[2][:, -self.config.action_horizon :].to(dtype=torch.float32)
        action_velocity = self.action_out_proj(action_output)
        visuotactile_velocity = self.visuotactile_out_proj(visuotactile_output)
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

        action_loss = self._action_flow_loss(
            action_velocity,
            action_target_velocity,
            training_phase,
        )
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
        # The tactile-only phase has no action objective. Joint training
        # restores action supervision, unfreezes the full model, and keeps the
        # future prediction objectives as auxiliary losses.
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
        prefix_embeddings, prefix_pad_mask, prefix_attention_ar = prefix
        prefix_attention_mask = self._prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_mask, prefix_attention_ar)
        )
        prefix_position_ids = torch.cumsum(prefix_pad_mask, dim=1) - 1
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None, None],
            use_cache=True,
            adarms_cond=[None, None, None],
        )

        actions = noise
        future_latents = future_noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            action_velocity, visuotactile_velocity = self.denoise_step(
                state,
                (prefix_pad_mask, past_key_values),
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
        if len(prefix) == 2:
            prefix_pad_mask, past_key_values = prefix
            return self._predict_vector_fields_with_prefix_cache(
                prefix_pad_mask,
                past_key_values,
                *action_suffix[:3],
                *visuotactile_suffix[:3],
                action_suffix[3],
                visuotactile_suffix[3],
            )
        return self._predict_vector_fields(
            *prefix,
            *action_suffix[:3],
            *visuotactile_suffix[:3],
            action_suffix[3],
            visuotactile_suffix[3],
        )


class PI0PrefixTactileExpertVisuoTactilePytorch(PI0ExpertVisuoTactilePytorch):
    """Variant whose current tactile prefix is processed by the tactile expert.

    The three transformer streams are:

    * PaliGemma: regular camera images and language only.
    * Action expert: state and noisy action suffix.
    * Tactile expert: current tactile prefix followed by the noisy future
      tactile suffix.

    Current tactile queries can only attend current tactile keys. Action and
    future tactile suffix queries may attend the current tactile prefix, while
    VL queries never attend it. Action and future tactile suffixes remain
    mutually isolated.
    """

    def __init__(self, config):
        super().__init__(config)
        vl_width = _gemma.get_config(config.paligemma_variant).width
        tactile_variant = _config_get(
            config,
            "visuotactile_expert_variant",
            _config_get(config, "visual_tactile_expert_variant", config.action_expert_variant),
        )
        tactile_width = _gemma.get_config(tactile_variant).width
        tactile_dtype = (
            self.paligemma_with_expert.gemma_visuotactile_expert.model.layers[
                0
            ].self_attn.q_proj.weight.dtype
        )
        self.tactile_prefix_proj = nn.Linear(
            vl_width,
            tactile_width,
            dtype=tactile_dtype,
        )

    def set_train_phase(self, phase: str) -> None:
        super().set_train_phase(phase)
        if phase == "latent":
            for parameter in self.tactile_prefix_proj.parameters():
                parameter.requires_grad_(True)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        observation=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return separate VL and current-tactile prefix streams."""
        vl_embeddings = []
        vl_pad_masks = []

        for image, image_mask in zip(images, img_masks, strict=True):
            image_embedding = self._apply_checkpoint(
                self.paligemma_with_expert.embed_image,
                image,
            )
            batch_size, token_count = image_embedding.shape[:2]
            vl_embeddings.append(image_embedding)
            vl_pad_masks.append(image_mask[:, None].expand(batch_size, token_count))

        def embed_language(tokens):
            embedding = self.paligemma_with_expert.embed_language_tokens(tokens)
            return embedding * math.sqrt(embedding.shape[-1])

        language_embedding = self._apply_checkpoint(embed_language, lang_tokens)
        vl_embeddings.append(language_embedding)
        vl_pad_masks.append(lang_masks)
        vl_embeddings = torch.cat(vl_embeddings, dim=1)
        vl_pad_masks = torch.cat(vl_pad_masks, dim=1).to(dtype=torch.bool)
        vl_attention_ar = torch.zeros_like(vl_pad_masks)

        tactile_embeddings = []
        tactile_pad_masks = []
        if observation is not None:
            for tactile, tactile_mask in self._iter_current_visuotactile(observation):
                tactile_embedding = self._embed_visuotactile_tokens(tactile)
                tactile_embedding = self._apply_checkpoint(
                    self.tactile_prefix_proj,
                    tactile_embedding,
                )
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
                tactile_embeddings.append(tactile_embedding)
                tactile_pad_masks.append(
                    token_mask.to(device=tactile_embedding.device, dtype=torch.bool)
                )

        if not tactile_embeddings:
            raise ValueError(
                "PI0PrefixTactileExpertVisuoTactilePytorch requires at least one "
                "current tactile input."
            )
        tactile_embeddings = torch.cat(tactile_embeddings, dim=1)
        tactile_pad_masks = torch.cat(tactile_pad_masks, dim=1)
        tactile_attention_ar = torch.zeros_like(tactile_pad_masks)
        return (
            vl_embeddings,
            vl_pad_masks,
            vl_attention_ar,
            tactile_embeddings,
            tactile_pad_masks,
            tactile_attention_ar,
        )

    @staticmethod
    def _make_prefix_tactile_attention_mask(
        pad_mask: Tensor,
        vl_len: int,
        action_len: int,
        tactile_prefix_len: int,
    ) -> Tensor:
        """Build the directed VL/action/current-tactile/future-tactile graph."""
        total_len = pad_mask.shape[1]
        tactile_start = vl_len + action_len
        future_start = tactile_start + tactile_prefix_len
        if not (0 <= vl_len <= tactile_start <= future_start <= total_len):
            raise ValueError("Invalid stream lengths for prefix-tactile attention.")

        allowed = torch.zeros(
            total_len,
            total_len,
            dtype=torch.bool,
            device=pad_mask.device,
        )
        # VL is completely independent from the tactile prefix.
        allowed[:vl_len, :vl_len] = True
        # Action sees normal VL context, itself, and current tactile.
        allowed[vl_len:tactile_start, :vl_len] = True
        allowed[vl_len:tactile_start, vl_len:tactile_start] = True
        allowed[vl_len:tactile_start, tactile_start:future_start] = True
        # Current tactile only sees current tactile.
        allowed[tactile_start:future_start, tactile_start:future_start] = True
        # Future tactile sees VL, current tactile, and itself, but not action.
        allowed[future_start:, :vl_len] = True
        allowed[future_start:, tactile_start:future_start] = True
        allowed[future_start:, future_start:] = True

        valid_pairs = pad_mask[:, :, None] & pad_mask[:, None, :]
        return allowed[None, :, :] & valid_pairs

    def _predict_vector_fields(
        self,
        vl_prefix,
        vl_pad_mask,
        _vl_attention_ar,
        tactile_prefix,
        tactile_prefix_pad_mask,
        _tactile_prefix_attention_ar,
        action_suffix,
        action_pad_mask,
        _action_attention_ar,
        visuotactile_suffix,
        visuotactile_pad_mask,
        _visuotactile_attention_ar,
        action_adarms_cond,
        visuotactile_adarms_cond,
    ) -> tuple[Tensor, Tensor]:
        target_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        vl_prefix = vl_prefix.to(dtype=target_dtype)
        action_suffix = action_suffix.to(dtype=target_dtype)
        tactile_prefix = tactile_prefix.to(dtype=target_dtype)
        visuotactile_suffix = visuotactile_suffix.to(dtype=target_dtype)
        tactile_stream = torch.cat([tactile_prefix, visuotactile_suffix], dim=1)

        pad_mask = torch.cat(
            [
                vl_pad_mask,
                action_pad_mask,
                tactile_prefix_pad_mask,
                visuotactile_pad_mask,
            ],
            dim=1,
        )
        attention_mask = self._make_prefix_tactile_attention_mask(
            pad_mask,
            vl_len=vl_pad_mask.shape[1],
            action_len=action_pad_mask.shape[1],
            tactile_prefix_len=tactile_prefix_pad_mask.shape[1],
        )
        attention_mask = self._prepare_attention_masks_4d(attention_mask)
        position_ids = torch.cumsum(pad_mask, dim=1) - 1

        def joint_forward(vl_tokens, action_tokens, tactile_tokens, mask, positions, action_cond, tactile_cond):
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[vl_tokens, action_tokens, tactile_tokens],
                use_cache=False,
                adarms_cond=[None, action_cond, tactile_cond],
            )
            return outputs[1], outputs[2]

        action_output, tactile_output = self._apply_checkpoint(
            joint_forward,
            vl_prefix,
            action_suffix,
            tactile_stream,
            attention_mask,
            position_ids,
            action_adarms_cond,
            visuotactile_adarms_cond,
        )
        action_output = action_output[:, -self.config.action_horizon :].float()
        future_output = tactile_output[:, -self.config.action_horizon :].float()
        return (
            self._apply_checkpoint(self.action_out_proj, action_output),
            self._apply_checkpoint(self.visuotactile_out_proj, future_output),
        )

    def _build_prefix_cache(
        self,
        vl_prefix: Tensor,
        vl_pad_mask: Tensor,
        tactile_prefix: Tensor,
        tactile_prefix_pad_mask: Tensor,
        action_len: int,
    ) -> DynamicCache:
        """Cache the independent VL and tactile prefixes, then concatenate K/V.

        The physical cache order is ``[VL, current tactile]``. Rotary positions
        still follow the full training layout ``[VL, action, tactile, future]``;
        consequently tactile positions reserve ``action_len`` positions between
        VL and tactile even though action is not part of the cache.
        """
        vl_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        tactile_dtype = self.paligemma_with_expert.gemma_visuotactile_expert.model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        vl_prefix = vl_prefix.to(dtype=vl_dtype)
        tactile_prefix = tactile_prefix.to(dtype=tactile_dtype)

        vl_attention_mask = self._prepare_attention_masks_4d(
            vl_pad_mask[:, :, None] & vl_pad_mask[:, None, :]
        )
        vl_position_ids = torch.cumsum(vl_pad_mask, dim=1) - 1
        _, vl_cache = self.paligemma_with_expert.forward(
            attention_mask=vl_attention_mask,
            position_ids=vl_position_ids,
            past_key_values=None,
            inputs_embeds=[vl_prefix, None, None],
            use_cache=True,
            adarms_cond=[None, None, None],
        )

        tactile_attention_mask = self._prepare_attention_masks_4d(
            tactile_prefix_pad_mask[:, :, None]
            & tactile_prefix_pad_mask[:, None, :]
        )
        tactile_position_ids = (
            torch.sum(vl_pad_mask, dim=1, keepdim=True)
            + action_len
            + torch.cumsum(tactile_prefix_pad_mask, dim=1)
            - 1
        )
        tactile_output = (
            self.paligemma_with_expert.gemma_visuotactile_expert.model.forward(
                attention_mask=tactile_attention_mask,
                position_ids=tactile_position_ids,
                past_key_values=None,
                inputs_embeds=tactile_prefix,
                use_cache=True,
                adarms_cond=None,
            )
        )
        tactile_cache = tactile_output.past_key_values

        combined_cache = DynamicCache()
        if len(vl_cache) != len(tactile_cache):
            raise ValueError(
                "VL and tactile prefix caches must have the same number of layers."
            )
        for layer_index in range(len(vl_cache)):
            vl_key, vl_value = vl_cache[layer_index]
            tactile_key, tactile_value = tactile_cache[layer_index]
            combined_cache.update(
                torch.cat([vl_key, tactile_key], dim=2),
                torch.cat([vl_value, tactile_value], dim=2),
                layer_index,
            )
        return combined_cache

    def _predict_vector_fields_with_prefix_cache(
        self,
        prefix_pad_masks,
        past_key_values,
        action_suffix,
        action_pad_mask,
        _action_attention_ar,
        visuotactile_suffix,
        visuotactile_pad_mask,
        _visuotactile_attention_ar,
        action_adarms_cond,
        visuotactile_adarms_cond,
    ) -> tuple[Tensor, Tensor]:
        """Predict suffixes using cached VL and current-tactile K/V."""
        vl_pad_mask, tactile_prefix_pad_mask = prefix_pad_masks
        action_dtype = self.paligemma_with_expert.gemma_expert.model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        tactile_dtype = self.paligemma_with_expert.gemma_visuotactile_expert.model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        action_suffix = action_suffix.to(dtype=action_dtype)
        visuotactile_suffix = visuotactile_suffix.to(dtype=tactile_dtype)

        # Cache keys are [VL, tactile prefix]; newly computed keys are
        # [action suffix, future tactile suffix].
        key_pad_mask = torch.cat(
            [
                vl_pad_mask,
                tactile_prefix_pad_mask,
                action_pad_mask,
                visuotactile_pad_mask,
            ],
            dim=1,
        )
        batch_size = key_pad_mask.shape[0]
        vl_len = vl_pad_mask.shape[1]
        tactile_len = tactile_prefix_pad_mask.shape[1]
        action_len = action_pad_mask.shape[1]
        future_len = visuotactile_pad_mask.shape[1]
        query_len = action_len + future_len

        allowed = torch.zeros(
            query_len,
            key_pad_mask.shape[1],
            dtype=torch.bool,
            device=key_pad_mask.device,
        )
        # Action queries -> cached VL, cached tactile, and action.
        allowed[:action_len, : vl_len + tactile_len + action_len] = True
        # Future tactile queries -> cached VL, cached tactile, and future
        # tactile, but never action.
        allowed[action_len:, : vl_len + tactile_len] = True
        allowed[action_len:, vl_len + tactile_len + action_len :] = True

        query_pad_mask = torch.cat(
            [action_pad_mask, visuotactile_pad_mask],
            dim=1,
        )
        valid_pairs = query_pad_mask[:, :, None] & key_pad_mask[:, None, :]
        attention_mask = self._prepare_attention_masks_4d(
            allowed[None, :, :] & valid_pairs
        )

        vl_valid_len = torch.sum(vl_pad_mask, dim=1, keepdim=True)
        action_position_ids = (
            vl_valid_len + torch.cumsum(action_pad_mask, dim=1) - 1
        )
        future_position_ids = (
            vl_valid_len
            + torch.sum(action_pad_mask, dim=1, keepdim=True)
            + torch.sum(tactile_prefix_pad_mask, dim=1, keepdim=True)
            + torch.cumsum(visuotactile_pad_mask, dim=1)
            - 1
        )
        position_ids = torch.cat(
            [action_position_ids, future_position_ids],
            dim=1,
        )

        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, action_suffix, visuotactile_suffix],
            use_cache=False,
            adarms_cond=[None, action_adarms_cond, visuotactile_adarms_cond],
        )
        action_output = outputs[1][:, -self.config.action_horizon :].float()
        future_output = outputs[2][:, -self.config.action_horizon :].float()
        return self.action_out_proj(action_output), self.visuotactile_out_proj(
            future_output
        )

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
        """Sample with independently built and merged VL/tactile prefix caches."""
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
        (
            vl_prefix,
            vl_pad_mask,
            _,
            tactile_prefix,
            tactile_prefix_pad_mask,
            _,
        ) = prefix

        # The action stream length is needed to preserve the same rotary
        # positions used by the full non-cached training layout.
        initial_time = torch.ones(batch_size, dtype=torch.float32, device=device)
        initial_action_suffix = self.embed_action_suffix(state, noise, initial_time)
        action_len = initial_action_suffix[1].shape[1]
        past_key_values = self._build_prefix_cache(
            vl_prefix,
            vl_pad_mask,
            tactile_prefix,
            tactile_prefix_pad_mask,
            action_len,
        )
        cached_prefix = (
            (vl_pad_mask, tactile_prefix_pad_mask),
            past_key_values,
        )

        actions = noise
        future_latents = future_noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            action_velocity, visuotactile_velocity = self.denoise_step(
                state,
                cached_prefix,
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


__all__ = [
    "IndependentTactileSiglipEncoder",
    "PI0ExpertVisuoTactilePytorch",
    "PI0PrefixTactileExpertVisuoTactilePytorch",
]
