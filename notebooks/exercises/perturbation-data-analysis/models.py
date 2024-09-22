"""This module defines a Multi-Layer Perceptron (MLP) model."""

import torch
from torch import nn

import pytorch_lightning as pl


class Encoder(nn.Module):
    """MLP encoder."""

    def __init__(self, in_features: int, latent_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=latent_dim),
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    """MLP decoder."""

    def __init__(self, latent_dim: int, out_features: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=out_features),
        )

    def forward(self, x):
        return self.l1(x)


class Autoencoder(pl.LightningModule):
    """Simple Autoencoder model."""

    def __init__(self, in_features: int, learning_rate: float = 1e-3) -> None:
        """
        Initialize the Autoencoder model.

        Args:
            in_features: The number of input features.
            learning_rate: The learning rate for the optimizer. Default is 1e-3.
        """
        super().__init__()

        self.encoder = Encoder(in_features=in_features, latent_dim=64)
        self.decoder = Decoder(latent_dim=64, out_features=in_features)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x):  # noqa: D102
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):  # noqa: D102
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_fn(input=x_hat, target=x)
        self.log(name="batch_idx", value=int(batch_idx), prog_bar=True)
        self.log(name="train_loss", value=loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):  # noqa: D102
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_fn(input=x_hat, target=x)
        self.log(name="test_loss", value=loss, prog_bar=True)

    def configure_optimizers(self):  # noqa: D102
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer
