from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch.pi0_expertvisuotactile_pytorch import PI0ExpertVisuoTactilePytorch


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

    model.future_visuotactile_autoencoder = nn.Linear(2, 2)
    model.action_in_proj = nn.Linear(2, 2)
    model.action_out_proj = nn.Linear(2, 2)
    model.visuotactile_in_proj = nn.Linear(2, 2)
    model.visuotactile_out_proj = nn.Linear(2, 2)
    model.visuotactile_time_mlp_in = nn.Linear(2, 2)
    model.visuotactile_time_mlp_out = nn.Linear(2, 2)
    return model


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


def test_latent_phase_only_unfreezes_tactile_expert_and_projections():
    model = make_phase_test_model()
    model.set_train_phase("latent")

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all(
        name.startswith(
            (
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
