import numpy as np
from openpi_client import action_chunk_broker
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _FakePytorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_calls = 0

    def sample_actions(self, device, observation, **kwargs):
        del device, kwargs
        self.sample_calls += 1
        return observation.state[:, None, :2].repeat(1, 3, 1)


def _make_model_ready_observation(value: float) -> dict:
    return {
        "image": {"camera": np.full((2, 2, 3), value, dtype=np.float32)},
        "image_mask": {"camera": np.True_},
        "state": np.array([value, value + 1, value + 2], dtype=np.float32),
    }


def test_infer_batch_calls_model_once():
    model = _FakePytorchModel()
    policy = _policy.Policy(model, is_pytorch=True, pytorch_device="cpu")

    results = policy.infer_batch([_make_model_ready_observation(1), _make_model_ready_observation(4)])

    assert model.sample_calls == 1
    assert len(results) == 2
    np.testing.assert_array_equal(results[0]["actions"], np.array([[1, 2], [1, 2], [1, 2]]))
    np.testing.assert_array_equal(results[1]["actions"], np.array([[4, 5], [4, 5], [4, 5]]))


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
