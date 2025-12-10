"""
chatter.models
==============

Neural network architectures for variational autoencoders.
"""

# Silence annoying warning
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*pkg_resources is deprecated as an API.*"
)

# Import necessary libraries
from typing import Optional, Tuple  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for a variational autoencoder operating on spectrograms.

    This encoder processes single-channel spectrogram inputs with a series of
    convolutional layers followed by batch normalization and Mish activation.
    It outputs the mean and log variance of the latent distribution.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    target_shape : tuple of int
        Shape of the input spectrogram as (height, width).
    """

    def __init__(self, latent_dim: int, target_shape: Tuple[int, int]) -> None:
        super().__init__()

        # Compute the spatial size of the feature map after four downsampling stages
        h, w = target_shape
        self.h_feat = h // 16
        self.w_feat = w // 16
        self.flat_size = 512 * self.h_feat * self.w_feat

        # Define convolutional blocks followed by batch normalization
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(self.flat_size, 1024)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, 1, height, width).

        Returns
        -------
        tuple of torch.Tensor
            A tuple (mu, log_var) representing the latent distribution parameters.
        """
        # Apply convolutional blocks with Mish activation
        x = F.mish(self.bn1(self.conv1(x)))
        x = F.mish(self.bn2(self.conv2(x)))
        x = F.mish(self.bn3(self.conv3(x)))
        x = F.mish(self.bn4(self.conv4(x)))

        # Flatten the feature map and project through a fully connected layer
        x = x.view(-1, self.flat_size)
        x = F.mish(self.fc(x))

        # Produce mean and log-variance vectors for the latent distribution
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var


class ConvDecoder(nn.Module):
    """
    Convolutional decoder using a resize-convolution architecture.

    This decoder reconstructs spectrogram images from latent vectors using
    nearest neighbor upsampling followed by convolution to mitigate
    checkerboard artifacts.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    target_shape : tuple of int
        Shape of the output spectrogram as (height, width).
    """

    def __init__(self, latent_dim: int, target_shape: Tuple[int, int]) -> None:
        super().__init__()

        # Compute the spatial size of the feature map that will be upsampled
        h, w = target_shape
        self.h_feat = h // 16
        self.w_feat = w // 16
        self.flat_size = 512 * self.h_feat * self.w_feat

        # Fully connected layers map the latent vector to a flattened feature map
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, self.flat_size)

        # Define three upsampling blocks and the final convolution
        self.upblock1 = self._make_up_block(512, 256)
        self.upblock2 = self._make_up_block(256, 128)
        self.upblock3 = self._make_up_block(128, 64)

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def _make_up_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create an upsampling block with batch normalization and Mish activation."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation with shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed spectrogram with shape (batch_size, 1, height, width).
        """
        # Project latent vector to the initial feature map
        x = F.mish(self.fc1(z))
        x = F.mish(self.fc2(x))
        x = x.view(-1, 512, self.h_feat, self.w_feat)

        # Apply upsampling blocks to reconstruct spatial resolution
        x = self.upblock1(x)
        x = self.upblock2(x)
        x = self.upblock3(x)

        # Produce reconstructed spectrogram in the [0, 1] range
        return torch.sigmoid(self.final_conv(x))


class VectorEncoder(nn.Module):
    """
    Fully connected encoder for a variational autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the flattened input vector.
    latent_dim : int
        Dimensionality of the latent space.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()

        # Two fully connected layers progressively reduce dimensionality
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Flattened input tensor with shape (batch_size, input_dim).

        Returns
        -------
        tuple of torch.Tensor
            A tuple (mu, log_var) representing the latent distribution parameters.
        """
        # Encode the input using Mish-activated fully connected layers
        x = F.mish(self.fc1(x))
        x = F.mish(self.fc2(x))

        return self.fc_mu(x), self.fc_log_var(x)


class VectorDecoder(nn.Module):
    """
    Fully connected decoder for an autoencoder.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation.
    output_dim : int
        Dimensionality of the flattened spectrogram output.
    """

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()

        # Three fully connected layers expand the latent code back to the output dimension
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation with shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed output with shape (batch_size, output_dim).
        """
        # Decode the latent representation using Mish-activated layers
        x = F.mish(self.fc1(z))
        x = F.mish(self.fc2(x))

        return torch.sigmoid(self.fc3(x))


class Encoder(nn.Module):
    """
    Unified autoencoder wrapper supporting both convolutional and vector architectures.

    Parameters
    ----------
    ae_type : str
        Type of autoencoder architecture ('convolutional' or 'vector').
    latent_dim : int
        Dimensionality of the latent space.
    input_dim : int, optional
        Dimensionality of the flattened input (required for 'vector').
    target_shape : tuple of int, optional
        Shape of the input spectrogram (required for 'convolutional').
    """

    def __init__(
        self,
        ae_type: str,
        latent_dim: int,
        input_dim: Optional[int] = None,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        # Store configuration about the chosen architecture and input shapes
        self.ae_type = ae_type
        self.input_dim = input_dim
        self.target_shape = target_shape

        # Build the appropriate encoder/decoder pair based on the architecture type
        if self.ae_type == "convolutional":
            if self.target_shape is None:
                raise ValueError(
                    "target_shape must be provided for convolutional architecture"
                )
            self.encoder = ConvEncoder(latent_dim, target_shape=self.target_shape)
            self.decoder = ConvDecoder(latent_dim, target_shape=self.target_shape)

        elif self.ae_type == "vector":
            if self.input_dim is None:
                raise ValueError(
                    "input_dim must be provided for the 'vector' architecture"
                )
            self.encoder = VectorEncoder(self.input_dim, latent_dim)
            self.decoder = VectorDecoder(latent_dim, self.input_dim)

        else:
            raise ValueError(f"Unknown ae_type: {ae_type}")

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Apply the reparameterization trick for a variational autoencoder.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution.
        log_var : torch.Tensor
            Log variance of the latent distribution.

        Returns
        -------
        torch.Tensor
            Sampled latent vector.
        """
        # Compute the standard deviation from the log-variance
        std = torch.exp(0.5 * log_var)
        # Sample noise from a standard normal distribution and apply the reparameterization trick
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an input tensor into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple of torch.Tensor
            A tuple (mu, log_var) representing the latent distribution parameters.
        """
        # Flatten input if using the vector architecture before encoding
        if self.ae_type == "vector":
            x = x.view(-1, self.input_dim)

        return self.encoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a full forward pass through the variational autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple of torch.Tensor
            A tuple (x_recon, mu, log_var).
        """
        # Preserve original input shape for potential reshaping
        original_shape = x.shape

        # Encode input, sample a latent vector, and decode it back to input space
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)

        # Reshape the output back to the original shape for vector architectures
        if self.ae_type == "vector":
            x_recon = x_recon.view(original_shape)

        return x_recon, mu, log_var


def ae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    fg_tau: float = 0.1,
    fg_alpha: float = 10.0,
) -> torch.Tensor:
    """
    Compute variational autoencoder loss with foreground weighting.

    This loss function combines a foreground-weighted L1 reconstruction term
    with a Kullback-Leibler divergence term. Foreground pixels are assigned
    higher weight to mitigate mode collapse on sparse spectrograms.

    Parameters
    ----------
    x : torch.Tensor
        Original input tensor.
    x_recon : torch.Tensor
        Reconstructed output tensor.
    mu : torch.Tensor
        Mean of the latent distribution.
    log_var : torch.Tensor
        Log variance of the latent distribution.
    beta : float, optional
        Weight for the KL divergence term. Default is 1.0.
    fg_tau : float, optional
        Threshold for identifying foreground pixels. Default is 0.1.
    fg_alpha : float, optional
        Weight multiplier for foreground pixels. Default is 10.0.

    Returns
    -------
    torch.Tensor
        Scalar loss value normalized by batch size.
    """
    # Start with uniform weights for all pixels
    weights = torch.ones_like(x)
    # Mark foreground pixels that exceed the threshold and upweight their contribution
    foreground_mask = x > fg_tau
    weights[foreground_mask] = fg_alpha

    # Compute per-pixel L1 reconstruction loss and apply foreground weights
    recon_loss_per_pixel = F.l1_loss(x_recon, x, reduction="none")
    weighted_recon_loss = torch.sum(weights * recon_loss_per_pixel)

    # Compute KL divergence between the approximate posterior and a unit Gaussian prior
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = weighted_recon_loss + beta * kld

    # Normalize the total loss by the batch size
    return total_loss / x.size(0)
