from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch.pi0_expertvisuotactile_pytorch import IndependentTactileSiglipEncoder
from openpi.models_pytorch.pi0_expertvisuotactile_pytorch import PI0ExpertVisuoTactilePytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class _VisionOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _TinyVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def forward(self, inputs):
        return _VisionOutput(self.projection(inputs))


class _TinyPaliGemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = _TinyVisionTower()
        self.multi_modal_projector = nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))

    def get_image_features(self, pixel_values):
        features = self.vision_tower(pixel_values).last_hidden_state
        features = self.multi_modal_projector(features)
        return features / (self.config.text_config.hidden_size**0.5)


def make_phase_test_model() -> PI0ExpertVisuoTactilePytorch:
    model = PI0ExpertVisuoTactilePytorch.__new__(PI0ExpertVisuoTactilePytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(future_joint_finetune_start_step=10)
    model.pretrained_autoencoder_loaded = True

    streams = nn.Module()
    streams.paligemma = nn.Linear(2, 2)
    streams.gemma_expert = nn.Linear(2, 2)
    streams.gemma_visuotactile_expert = nn.Linear(2, 2)
    model.paligemma_with_expert = streams

    model.tactile_encoder = nn.Linear(2, 2)
    model.future_visuotactile_autoencoder = nn.Linear(2, 2)
    model.action_in_proj = nn.Linear(2, 2)
    model.action_out_proj = nn.Linear(2, 2)
    model.visuotactile_in_proj = nn.Linear(2, 2)
    model.visuotactile_out_proj = nn.Linear(2, 2)
    model.visuotactile_time_mlp_in = nn.Linear(2, 2)
    model.visuotactile_time_mlp_out = nn.Linear(2, 2)
    return model


def test_independent_tactile_encoder_matches_runtime_image_features_then_diverges():
    image_encoder = _TinyPaliGemmaModel()
    tactile_encoder = IndependentTactileSiglipEncoder(image_encoder)
    inputs = torch.randn(2, 5, 3)

    assert torch.allclose(image_encoder.get_image_features(inputs), tactile_encoder(inputs))
    assert (
        image_encoder.vision_tower.projection.weight.data_ptr()
        != tactile_encoder.vision_tower.projection.weight.data_ptr()
    )

    tactile_encoder(inputs).sum().backward()
    assert tactile_encoder.vision_tower.projection.weight.grad is not None
    assert image_encoder.vision_tower.projection.weight.grad is None

    with torch.no_grad():
        tactile_encoder.vision_tower.projection.weight.add_(1)

    assert not torch.allclose(image_encoder.get_image_features(inputs), tactile_encoder(inputs))


def test_loading_base_weights_initializes_independent_tactile_encoder_after_image_encoder():
    model = PI0ExpertVisuoTactilePytorch.__new__(PI0ExpertVisuoTactilePytorch)
    nn.Module.__init__(model)

    streams = nn.Module()
    streams.paligemma = _TinyPaliGemmaModel()
    model.paligemma_with_expert = streams
    model.tactile_encoder = IndependentTactileSiglipEncoder(streams.paligemma)

    base_state = {
        key: torch.full_like(value, 7.0)
        for key, value in model.state_dict().items()
        if not key.startswith("tactile_encoder.")
    }
    model.load_state_dict(base_state, strict=False)

    image_state = model.paligemma_with_expert.paligemma.vision_tower.state_dict()
    tactile_state = model.tactile_encoder.vision_tower.state_dict()
    assert image_state.keys() == tactile_state.keys()
    assert all(torch.equal(image_state[key], tactile_state[key]) for key in image_state)


def test_training_phase_schedule_requires_pretrained_autoencoder():
    model = make_phase_test_model()

    assert model.get_training_phase(0) == "latent"
    assert model.get_training_phase(9) == "latent"
    assert model.get_training_phase(10) == "joint"

    model.pretrained_autoencoder_loaded = False
    with pytest.raises(RuntimeError, match="pretrained autoencoder"):
        model.get_training_phase(0)


def test_training_phase_defaults_to_joint_after_5000_steps():
    model = make_phase_test_model()
    model.config = SimpleNamespace()

    assert model.get_training_phase(4_999) == "latent"
    assert model.get_training_phase(5_000) == "joint"


def test_future_loss_weights_change_after_latent_phase():
    model = make_phase_test_model()
    model.future_flow_loss_weight = 1.0
    model.future_visuotactile_loss_weight = 1.0
    model.future_joint_flow_loss_weight = 0.05
    model.future_joint_visuotactile_loss_weight = 0.1
    model.future_joint_autoencoder_loss_weight = 0.05

    assert model._future_loss_weights("latent") == (1.0, 1.0, 0.0)  # noqa: SLF001
    assert model._future_loss_weights("joint") == (0.05, 0.1, 0.05)  # noqa: SLF001


def test_action_loss_is_disabled_during_tactile_only_phase():
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn_like(prediction)

    latent_loss = PI0ExpertVisuoTactilePytorch._action_flow_loss(  # noqa: SLF001
        prediction,
        target,
        "latent",
    )
    joint_loss = PI0ExpertVisuoTactilePytorch._action_flow_loss(  # noqa: SLF001
        prediction,
        target,
        "joint",
    )

    assert torch.equal(latent_loss, torch.zeros(2, 3))
    assert not latent_loss.requires_grad
    assert joint_loss.requires_grad
    assert joint_loss.shape == (2, 3)


def test_latent_phase_only_unfreezes_tactile_expert_and_projections():
    model = make_phase_test_model()
    model.set_train_phase("latent")

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all(
        name.startswith(
            (
                "tactile_encoder.",
                "paligemma_with_expert.gemma_visuotactile_expert.",
                "visuotactile_in_proj.",
                "visuotactile_out_proj.",
                "visuotactile_time_mlp_in.",
                "visuotactile_time_mlp_out.",
            )
        )
        for name in trainable_names
    )
    assert not any(name.startswith("future_visuotactile_autoencoder.") for name in trainable_names)
    assert not any(name.startswith("paligemma_with_expert.gemma_expert.") for name in trainable_names)
    assert not any(name.startswith("paligemma_with_expert.paligemma.") for name in trainable_names)

    model.set_train_phase("joint")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_tactile_expert_can_initialize_from_action_expert():
    model = make_phase_test_model()
    with torch.no_grad():
        model.paligemma_with_expert.gemma_expert.weight.fill_(2.0)
        model.paligemma_with_expert.gemma_expert.bias.fill_(3.0)
        model.paligemma_with_expert.gemma_visuotactile_expert.weight.zero_()
        model.paligemma_with_expert.gemma_visuotactile_expert.bias.zero_()

    model.initialize_tactile_expert_from_action_expert()

    action_state = model.paligemma_with_expert.gemma_expert.state_dict()
    tactile_state = model.paligemma_with_expert.gemma_visuotactile_expert.state_dict()
    assert action_state.keys() == tactile_state.keys()
    assert all(torch.equal(action_state[key], tactile_state[key]) for key in action_state)


def test_tactile_suffix_uses_bidirectional_self_attention():
    model = PI0ExpertVisuoTactilePytorch.__new__(PI0ExpertVisuoTactilePytorch)
    nn.Module.__init__(model)
    model.pi05 = True
    model.gradient_checkpointing_enabled = False
    model.visuotactile_in_proj = nn.Linear(2, 4)
    model.visuotactile_time_mlp_in = nn.Linear(4, 4)
    model.visuotactile_time_mlp_out = nn.Linear(4, 4)

    _, _, attention_ar, _ = model.embed_visuotactile_suffix(
        torch.randn(1, 3, 2),
        torch.tensor([0.5]),
    )

    assert attention_ar[0].tolist() == [0, 0, 0]


def test_action_and_future_tactile_attention_are_mutually_isolated():
    # [prefix, action x2, tactile x2]
    attention_ar = torch.tensor([[0, 1, 0, 0, 0]])
    attention_mask = make_att_2d_masks(
        torch.ones_like(attention_ar, dtype=torch.bool),
        attention_ar,
    )
    attention_mask = PI0ExpertVisuoTactilePytorch._isolate_action_and_tactile_attention(  # noqa: SLF001
        attention_mask,
        action_start=1,
        tactile_start=3,
    )

    assert not attention_mask[0, 1:3, 3:5].any()  # action -> tactile
    assert not attention_mask[0, 3:5, 1:3].any()  # tactile -> action
    assert attention_mask[0, 1:3, 0].all()  # action -> prefix
    assert attention_mask[0, 3:5, 0].all()  # tactile -> prefix
    assert attention_mask[0, 1:3, 1:3].all()  # action -> action
    assert attention_mask[0, 3:5, 3:5].all()  # tactile -> tactile
