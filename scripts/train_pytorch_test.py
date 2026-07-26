import os

import pytest
import torch

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import pi0_config
from openpi.training import config as _config

from . import train_pytorch


class _DummyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = torch.nn.Linear(1, 1)
        self.gemma_expert = torch.nn.Linear(1, 1)
        self.gemma_visuotactile_expert = torch.nn.Linear(1, 1)


class _DummyExpertVisuoTactileModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _DummyExperts()
        self.action_in_proj = torch.nn.Linear(1, 1)
        self.action_out_proj = torch.nn.Linear(1, 1)
        self.time_mlp_in = torch.nn.Linear(1, 1)
        self.tactile_encoder = torch.nn.Linear(1, 1)
        self.visuotactile_in_proj = torch.nn.Linear(1, 1)
        self.future_visuotactile_autoencoder = torch.nn.Linear(1, 1)


def test_expert_visuotactile_optimizer_learning_rates():
    model = _DummyExpertVisuoTactileModel()
    model_cfg = pi0_config.Pi0VisuoTactileConfig(
        pi05=True,
        use_separate_visuotactile_expert=True,
        use_separate_tactile_encoder=True,
        future_head_lr_multiplier=2.0,
        future_backbone_lr_multiplier=0.25,
    )

    groups = train_pytorch.build_optimizer_param_groups(model, model_cfg, peak_lr=5e-5)
    group_by_name = {group["name"]: group for group in groups}
    parameter_group = {id(parameter): group["name"] for group in groups for parameter in group["params"]}
    named_parameter_group = {name: parameter_group[id(parameter)] for name, parameter in model.named_parameters()}

    assert groups[0]["name"] == "action_expert"
    assert named_parameter_group["paligemma_with_expert.paligemma.weight"] == "backbone"
    assert named_parameter_group["paligemma_with_expert.gemma_expert.weight"] == "action_expert"
    assert named_parameter_group["action_in_proj.weight"] == "action_expert"
    assert named_parameter_group["action_out_proj.weight"] == "action_expert"
    assert named_parameter_group["tactile_encoder.weight"] == "tactile_encoder"
    assert named_parameter_group["visuotactile_in_proj.weight"] == "tactile_expert"
    assert named_parameter_group["future_visuotactile_autoencoder.weight"] == "future_autoencoder"

    assert group_by_name["action_expert"]["lr"] == pytest.approx(5e-5)
    assert group_by_name["backbone"]["lr"] == pytest.approx(5e-5)
    assert group_by_name["tactile_encoder"]["lr"] == pytest.approx(5e-5)
    assert group_by_name["tactile_expert"]["lr"] == pytest.approx(1e-4)
    assert group_by_name["future_autoencoder"]["lr"] == pytest.approx(1.25e-5)


def test_expert_visuotactile_libero_uses_pi05_libero_policy_lr():
    config = _config.get_config("pi05_expert_visuotactile_libero")

    assert config.lr_schedule.warmup_steps == 1_000
    assert config.lr_schedule.peak_lr == pytest.approx(5e-5)
    assert config.lr_schedule.decay_steps == 1_000_000
    assert config.lr_schedule.decay_lr == pytest.approx(5e-5)
    assert config.model.future_head_lr_multiplier == pytest.approx(2.0)
    assert config.model.future_backbone_lr_multiplier == pytest.approx(0.25)
