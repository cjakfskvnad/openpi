import torch

from openpi.models_pytorch.pi0_visuotactile_pytorch import FutureVisuoTactileVisionAutoencoder
from openpi.models_pytorch.pi0_visuotactile_pytorch import _reconstruction_loss_terms
from openpi.models_pytorch.pi0_visuotactile_pytorch import _temporal_difference_loss


def make_autoencoder() -> FutureVisuoTactileVisionAutoencoder:
    return FutureVisuoTactileVisionAutoencoder(
        encoder_width=32,
        latent_dim=8,
        future_shape=(32, 32, 3),
        decoder_width=64,
        decoder_depth=1,
        latent_grid_size=(4, 4),
    )


def test_spatial_autoencoder_shapes_and_gradients():
    autoencoder = make_autoencoder()
    target = torch.rand(2, 3, 32, 32, 3) * 2.0 - 1.0

    latents = autoencoder.encode(target)
    reconstruction = autoencoder.decode(latents)
    loss = _reconstruction_loss_terms(reconstruction, target)["charbonnier"].mean()
    loss.backward()

    assert latents.shape == (2, 3, 8 * 4 * 4)
    assert reconstruction.shape == target.shape
    assert torch.allclose(latents.mean(dim=-1), torch.zeros(2, 3), atol=1e-5)
    assert autoencoder.encoder_out[-1].weight.grad is not None
    assert autoencoder.decoder_out[-1].weight.grad is not None


def test_reconstruction_and_temporal_losses_are_zero_for_identical_images():
    images = torch.rand(2, 3, 3, 16, 16) * 2.0 - 1.0

    terms = _reconstruction_loss_terms(images, images)
    temporal = _temporal_difference_loss(images, images)

    assert torch.allclose(terms["mse"], torch.zeros(2, 3))
    assert torch.allclose(terms["gradient"], torch.zeros(2, 3))
    assert torch.allclose(terms["pyramid"], torch.zeros(2, 3))
    assert torch.allclose(temporal, torch.zeros(2, 3))
