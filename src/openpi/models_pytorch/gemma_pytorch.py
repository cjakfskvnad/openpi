from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        action_adarms_cond_dim=None,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=(
                action_adarms_cond_dim or action_expert_config.width
            )
            if use_adarms[1]
            else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,  # noqa: FBT001
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PaliGemmaWithActionAndVisuoTactileExpertsModel(PaliGemmaWithExpertModel):
    """PaliGemma with independent action and future visuo-tactile experts.

    The three token streams use their own projections, MLPs, and normalization
    parameters while sharing one attention operation at every transformer
    layer. Consequently, the expert widths may differ, but their attention
    depth, head count, key/value head count, and head dimension must match.

    A prefix-only pass can cache PaliGemma-projected keys and values. Subsequent
    multi-stream passes may read that cache while jointly computing the action
    and visuo-tactile suffixes, but they cannot update it because both suffixes
    change at every denoising step.
    """

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        visuotactile_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        action_adarms_cond_dim=None,
    ):
        if use_adarms is None:
            use_adarms = [False, False, False]
        if len(use_adarms) != 3:
            raise ValueError(f"use_adarms must contain three entries, got {len(use_adarms)}.")

        super().__init__(
            vlm_config,
            action_expert_config,
            use_adarms=use_adarms[:2],
            action_adarms_cond_dim=action_adarms_cond_dim,
            precision=precision,
        )

        self._validate_expert_attention_configs(
            vlm_config,
            action_expert_config,
            visuotactile_expert_config,
        )
        visuotactile_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=visuotactile_expert_config.head_dim,
            hidden_size=visuotactile_expert_config.width,
            intermediate_size=visuotactile_expert_config.mlp_dim,
            num_attention_heads=visuotactile_expert_config.num_heads,
            num_hidden_layers=visuotactile_expert_config.depth,
            num_key_value_heads=visuotactile_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[2],
            adarms_cond_dim=visuotactile_expert_config.width if use_adarms[2] else None,
        )
        self.gemma_visuotactile_expert = GemmaForCausalLM(config=visuotactile_expert_config_hf)
        self.gemma_visuotactile_expert.model.embed_tokens = None

        # The parent applies precision before this expert exists, so apply it
        # once more after registering the new module.
        self.to_bfloat16_for_selected_params(precision)

    @staticmethod
    def _validate_expert_attention_configs(*configs):
        shared_fields = ("depth", "num_heads", "num_kv_heads", "head_dim")
        for field in shared_fields:
            values = [getattr(config, field) for config in configs]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"All experts must share {field}; got {values}.")

    def _models(self):
        return [
            self.paligemma.language_model,
            self.gemma_expert.model,
            self.gemma_visuotactile_expert.model,
        ]

    def _forward_single_stream(
        self,
        stream_index,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        adarms_cond,
    ):
        output = self._models()[stream_index].forward(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )
        outputs = [None, None, None]
        outputs[stream_index] = output.last_hidden_state
        prefix_past_key_values = output.past_key_values if stream_index == 0 else None
        return outputs, prefix_past_key_values

    def _compute_joint_layer(
        self,
        layer_idx,
        hidden_streams,
        attention_mask,
        position_ids,
        adarms_cond,
        past_key_values=None,
    ):
        models = self._models()
        active_indices = [index for index, hidden in enumerate(hidden_streams) if hidden is not None]

        query_states = []
        key_states = []
        value_states = []
        gates = {}
        for index in active_indices:
            layer = models[index].layers[layer_idx]
            normalized, gate = layer.input_layernorm(hidden_streams[index], cond=adarms_cond[index])
            gates[index] = gate

            input_shape = normalized.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_states.append(layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2))
            key_states.append(layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2))
            value_states.append(layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2))

        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)

        dummy_tensor = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = self.paligemma.language_model.rotary_emb(dummy_tensor, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,
        )
        if past_key_values is not None:
            prefix_key_states, prefix_value_states = past_key_values[layer_idx]
            key_states = torch.cat([prefix_key_states, key_states], dim=2)
            value_states = torch.cat([prefix_value_states, value_states], dim=2)

        reference_attention = self.paligemma.language_model.layers[layer_idx].self_attn
        attention_output, _ = modeling_gemma.eager_attention_forward(
            reference_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            reference_attention.scaling,
            dropout=0.0 if not self.training else reference_attention.attention_dropout,
        )

        outputs = [None, None, None]
        start_pos = 0
        for index in active_indices:
            hidden_states = hidden_streams[index]
            layer = models[index].layers[layer_idx]
            end_pos = start_pos + hidden_states.shape[1]
            stream_attention = attention_output[:, start_pos:end_pos].reshape(
                hidden_states.shape[0],
                hidden_states.shape[1],
                layer.self_attn.q_proj.out_features,
            )
            stream_attention = stream_attention.to(dtype=layer.self_attn.o_proj.weight.dtype)
            projected_attention = layer.self_attn.o_proj(stream_attention)

            output = modeling_gemma._gated_residual(  # noqa: SLF001
                hidden_states,
                projected_attention,
                gates[index],
            )
            first_residual = output
            output, gate = layer.post_attention_layernorm(output, cond=adarms_cond[index])
            output = output.to(dtype=layer.mlp.up_proj.weight.dtype)
            output = layer.mlp(output)
            outputs[index] = modeling_gemma._gated_residual(first_residual, output, gate)  # noqa: SLF001
            start_pos = end_pos

        return outputs

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,  # noqa: FBT001
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if inputs_embeds is None or len(inputs_embeds) != 3:
            raise ValueError("inputs_embeds must be [prefix, action, visuotactile].")
        if adarms_cond is None:
            adarms_cond = [None, None, None]
        if len(adarms_cond) != 3:
            raise ValueError("adarms_cond must contain prefix, action, and visuotactile entries.")

        active_indices = [index for index, hidden in enumerate(inputs_embeds) if hidden is not None]
        if not active_indices:
            raise ValueError("At least one input stream must be provided.")
        if len(active_indices) == 1:
            index = active_indices[0]
            return self._forward_single_stream(
                index,
                inputs_embeds[index],
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                adarms_cond[index],
            )
        if use_cache:
            raise ValueError("Updating a KV cache is only supported for single-stream forward passes.")
        if past_key_values is not None and 0 in active_indices:
            raise ValueError("A prefix KV cache can only be used when the prefix input stream is omitted.")

        models = self._models()
        use_gradient_checkpointing = self.training and any(
            getattr(models[index], "gradient_checkpointing", False) for index in active_indices
        )

        hidden_streams = list(inputs_embeds)
        for layer_idx in range(self.paligemma.config.text_config.num_hidden_layers):
            if use_gradient_checkpointing:

                def checkpointed_layer(*active_hidden_states, current_layer_idx=layer_idx):
                    checkpoint_inputs = [None, None, None]
                    for index, hidden in zip(active_indices, active_hidden_states, strict=True):
                        checkpoint_inputs[index] = hidden
                    checkpoint_outputs = self._compute_joint_layer(
                        current_layer_idx,
                        checkpoint_inputs,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        past_key_values,
                    )
                    return tuple(checkpoint_outputs[index] for index in active_indices)

                active_outputs = torch.utils.checkpoint.checkpoint(
                    checkpointed_layer,
                    *(hidden_streams[index] for index in active_indices),
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if len(active_indices) == 1:
                    active_outputs = (active_outputs,)
                for index, output in zip(active_indices, active_outputs, strict=True):
                    hidden_streams[index] = output
            else:
                hidden_streams = self._compute_joint_layer(
                    layer_idx,
                    hidden_streams,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    past_key_values,
                )

        outputs = [None, None, None]
        for index in active_indices:
            outputs[index], _ = models[index].norm(hidden_streams[index], cond=adarms_cond[index])
        return outputs, None
