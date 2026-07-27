import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0VisuoTactileConfig(Pi0Config):
    """Pi0 PyTorch-only config for visuo-tactile current inputs and future prediction."""

    # Use a third Gemma stream dedicated to future latent prediction instead
    # of concatenating future latents into the action expert.
    use_separate_visuotactile_expert: bool = False
    # Deprecated compatibility flag. The separate-expert PyTorch model always
    # uses an independent tactile SigLIP encoder.
    use_separate_tactile_encoder: bool = False
    # How continuous robot state conditions the action expert. "none" keeps
    # pi0.5's original behavior; "adarms" injects an encoded state through
    # the action expert's adaptive RMSNorm layers.
    state_input_mode: str = "none"
    # Cross-attention between the action and future visuo-tactile suffixes.
    # "isolated" preserves the original two-way isolation,
    # "future_attends_action" lets future queries read action keys only, and
    # "bidirectional" enables both cross-stream directions.
    action_visuotactile_attention_mode: str = "isolated"
    visuotactile_expert_variant: _gemma.Variant = "gemma_300m"
    visuotactile_keys: tuple[str, ...] = ("visuotactile_0_rgb", "tactile_0_rgb")
    future_visuotactile_key: str = "future_visuotactile"
    future_visuotactile_shape: tuple[int, ...] | None = None
    # Number of channels in the spatial latent grid. The flattened flow state
    # has latent_dim * grid_height * grid_width values per future frame.
    future_visuotactile_latent_dim: int = 16
    future_visuotactile_latent_grid_size: tuple[int, int] = (14, 14)
    future_visuotactile_encoder_width: int = 64
    future_visuotactile_decoder_width: int = 512
    future_visuotactile_decoder_depth: int = 2
    future_visuotactile_decoder_num_heads: int = 8
    future_visuotactile_patch_size: int = 16
    future_visuotactile_encoder_chunk_size: int = 8
    # Preserve the latent grid as spatiotemporal tokens instead of flattening
    # each complete future frame into one token. A token contains one
    # token_patch_size x token_patch_size patch of the latent grid.
    use_spatiotemporal_future_tokens: bool = False
    future_visuotactile_token_patch_size: int = 2
    action_loss_weight: float = 1.0
    future_flow_loss_weight: float = 1.0
    future_visuotactile_loss_weight: float = 1.0
    future_autoencoder_loss_weight: float = 1.0
    # After future_joint_finetune_start_step, keep action loss at full weight
    # and reduce auxiliary future-prediction objectives.
    future_joint_flow_loss_weight: float = 0.05
    future_joint_visuotactile_loss_weight: float = 0.1
    future_joint_autoencoder_loss_weight: float = 0.05
    future_mse_loss_weight: float = 0.1
    future_charbonnier_loss_weight: float = 1.0
    future_ssim_loss_weight: float = 0.2
    future_gradient_loss_weight: float = 0.1
    future_pyramid_loss_weight: float = 0.1
    future_temporal_loss_weight: float = 0.2
    # Keep the pretrained future-image codec fixed during every training
    # phase. Reconstruction gradients can still pass through its decoder to
    # the future expert while codec parameters remain frozen.
    freeze_future_autoencoder: bool = False

    # Staged expert PyTorch training: tactile-only flow matching before this
    # boundary, then full-model joint action and future prediction training.
    future_autoencoder_pretrain_steps: int = 5_000
    future_joint_finetune_start_step: int = 5_000
    future_head_lr_multiplier: float = 4.0
    future_backbone_lr_multiplier: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if self.state_input_mode not in {"none", "adarms"}:
            raise ValueError(f"state_input_mode must be one of {{'none', 'adarms'}}, got {self.state_input_mode!r}.")
        valid_attention_modes = {
            "isolated",
            "future_attends_action",
            "bidirectional",
        }
        if self.action_visuotactile_attention_mode not in valid_attention_modes:
            raise ValueError(
                "action_visuotactile_attention_mode must be one of "
                f"{sorted(valid_attention_modes)}, "
                f"got {self.action_visuotactile_attention_mode!r}."
            )
        if self.future_visuotactile_token_patch_size <= 0:
            raise ValueError(
                "future_visuotactile_token_patch_size must be positive, "
                f"got {self.future_visuotactile_token_patch_size}."
            )
        if self.use_spatiotemporal_future_tokens:
            grid_height, grid_width = self.future_visuotactile_latent_grid_size
            patch_size = self.future_visuotactile_token_patch_size
            if grid_height % patch_size or grid_width % patch_size:
                raise ValueError(
                    "future_visuotactile_latent_grid_size must be divisible by "
                    "future_visuotactile_token_patch_size when spatiotemporal "
                    f"tokens are enabled; got grid {(grid_height, grid_width)} "
                    f"and patch size {patch_size}."
                )
