"""Autoencoder model."""

import lightning as l
import torch
import torch.nn as nn
import torch.nn.functional as f


class Encoder(nn.Module):
    """MLP encoder."""

    def __init__(self, latent_dim: int) -> None:  # noqa: D107
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=latent_dim),
        )

    def forward(self, x):  # noqa: D102
        return self.l1(x)


class Decoder(nn.Module):
    """MLP decoder."""

    def __init__(self, latent_dim: int) -> None:  # noqa: D107
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=28 * 28),
        )

    def forward(self, x):  # noqa: D102
        return self.l1(x)


class AutoEncoder(l.LightningModule):
    """Autoencoder."""

    def __init__(self, lr: float, latent_dim: int) -> None:  # noqa: D107
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.lr = lr

    def training_step(self, batch):  # noqa: D102
        x, _ = batch
        x = x.view(x.size(0), -1)  # Flatten the input image.
        z = self.encoder(x)
        x_hat = self.decoder(z)
        train_loss = f.mse_loss(input=x_hat, target=x)
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return train_loss

    def test_step(self, batch):  # noqa: D102
        x, _ = batch
        x = x.view(x.size(0), -1)  # Flatten the input image.
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = f.mse_loss(input=x_hat, target=x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):  # noqa: D102
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer
