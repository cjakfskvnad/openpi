import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def test_visuotactile_inputs_use_strictly_future_frames():
    head_images = np.stack([np.full((8, 8, 3), fill_value=frame_index, dtype=np.uint8) for frame_index in range(11)])
    data = {
        "observation/image": head_images,
        "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
    }

    transformed = libero_policy.LiberoVisuoTactileInputs(model_type=_model.ModelType.PI05)(data)

    assert transformed["future_visuotactile"].shape == (10, 224, 224, 3)
    assert np.all(transformed["image"]["base_0_rgb"] == 0)
    assert np.isclose(transformed["future_visuotactile"][0, 112, 112, 0], 1 / 255 * 2 - 1)
    assert np.isclose(transformed["future_visuotactile"][-1, 112, 112, 0], 10 / 255 * 2 - 1)
    assert "visuotactile_0_rgb" not in transformed["image"]
