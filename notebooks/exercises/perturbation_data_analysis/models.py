"""This module defines an autoencoder model."""

import pytorch_lightning as pl
import torch
from torch import nn


class Encoder(nn.Module):
    """MLP encoder.

    Args:
        in_features: The number of input features.
        latent_dim: Dimension of the latent space.
    """

    def __init__(self, in_features: int, latent_dim: int) -> None:
        """Initialize the encoder."""
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.

        Args:
            x: The input tensor.

        Returns:
            The encoded tensor.
        """
        return self.l1(x)


class Decoder(nn.Module):
    """MLP decoder.

    Args:
        latent_dim: Dimension of the latent space.
        out_features: The number of output features.
    """

    def __init__(self, latent_dim: int, out_features: int) -> None:
        """Initialize the decoder."""
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x: The input tensor.

        Returns:
            The encoded tensor.
        """
        return self.l1(x)


class Autoencoder(pl.LightningModule):
    """Simple Autoencoder model.

    Args:
        in_features: The number of input features.
        learning_rate: The learning rate for the optimizer.
    """

    def __init__(self, in_features: int, learning_rate: float = 1e-3) -> None:
        """Initialize the autoencoder."""
        super().__init__()
        self.encoder = Encoder(in_features=in_features, latent_dim=64)
        self.decoder = Decoder(latent_dim=64, out_features=in_features)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the autoencoder.

        Args:
            x: The input tensor.

        Returns:
            The reconstructed tensor.
        """
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):  # noqa: D102
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_fn(input=x_hat, target=x)

        # Log batch index and training loss.
        self.log(name="batch_idx", value=int(batch_idx), prog_bar=True)
        self.log(name="train_loss", value=loss, prog_bar=True)

        return loss

    def test_step(self, batch):  # noqa: D102
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_fn(input=x_hat, target=x)

        # Log test loss.
        self.log(name="test_loss", value=loss, prog_bar=True)

    def configure_optimizers(self):  # noqa: D102
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer
