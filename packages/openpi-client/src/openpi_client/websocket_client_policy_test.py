import numpy as np

from openpi_client import msgpack_numpy
from openpi_client import websocket_api
from openpi_client import websocket_client_policy


class _FakeWebsocket:
    def __init__(self, response):
        self._response = msgpack_numpy.packb(response)
        self.sent = None

    def send(self, data):
        self.sent = data

    def recv(self):
        return self._response


def test_infer_batch_uses_batch_protocol():
    expected = [{"actions": np.array([[1.0]])}, {"actions": np.array([[2.0]])}]
    websocket = _FakeWebsocket(
        {
            websocket_api.BATCH_RESPONSE_KEY: expected,
            "server_timing": {"infer_ms": 10.0},
        }
    )
    client = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()  # noqa: SLF001
    client._ws = websocket  # noqa: SLF001
    observations = [{"state": np.array([1.0])}, {"state": np.array([2.0])}]

    actual = client.infer_batch(observations)

    request = msgpack_numpy.unpackb(websocket.sent)
    assert len(request[websocket_api.BATCH_REQUEST_KEY]) == 2
    np.testing.assert_array_equal(actual[0]["actions"], expected[0]["actions"])
    np.testing.assert_array_equal(actual[1]["actions"], expected[1]["actions"])
    assert actual[0]["server_timing"] == {"infer_ms": 10.0}
