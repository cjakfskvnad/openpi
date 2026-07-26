from openpi_client import base_policy
from openpi_client import websocket_api

from openpi.serving import websocket_policy_server


class _BatchPolicy(base_policy.BasePolicy):
    def __init__(self):
        self.infer_calls = 0
        self.batch_calls = 0

    def infer(self, obs):
        self.infer_calls += 1
        return {"value": obs["value"]}

    def infer_batch(self, observations):
        self.batch_calls += 1
        return [{"value": obs["value"]} for obs in observations]


def test_batch_request_uses_policy_batch_method():
    policy = _BatchPolicy()
    request = {websocket_api.BATCH_REQUEST_KEY: [{"value": 1}, {"value": 2}]}

    response = websocket_policy_server._infer_request(policy, request)  # noqa: SLF001

    assert policy.infer_calls == 0
    assert policy.batch_calls == 1
    assert response == {websocket_api.BATCH_RESPONSE_KEY: [{"value": 1}, {"value": 2}]}


def test_single_request_remains_compatible():
    policy = _BatchPolicy()

    response = websocket_policy_server._infer_request(policy, {"value": 1})  # noqa: SLF001

    assert policy.infer_calls == 1
    assert policy.batch_calls == 0
    assert response == {"value": 1}
