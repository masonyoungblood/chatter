import torch
from chatter import Trainer


def test_trainer_forward_pass(tiny_config):
    # Smoke test: one forward pass through the VAE with tiny shapes
    trainer = Trainer(tiny_config)
    x = torch.randn(
        2, 1, tiny_config["target_shape"][0], tiny_config["target_shape"][1]
    ).to(trainer.device)
    recon, mu, log_var = trainer.ae_model(x)
    assert recon.shape == x.shape
    assert mu.shape[0] == x.shape[0]
    assert log_var.shape == mu.shape


def test_ae_loss_returns_scalar(tiny_config):
    from chatter.models import ae_loss

    # Loss should be a finite scalar for a small batch
    trainer = Trainer(tiny_config)
    x = torch.randn(
        2, 1, tiny_config["target_shape"][0], tiny_config["target_shape"][1]
    ).to(trainer.device)
    recon, mu, log_var = trainer.ae_model(x)
    loss = ae_loss(x, recon, mu, log_var)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
