"""This module defines a Variational Autoencoder (VAE) model."""

import pytorch_lightning as pl
import torch
from torch import nn


class VAE(pl.LightningModule):
    """Simple Variational Autoencoder model."""

    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        learning_rate: float = 1e-3,
        loss_type: str = "mse_kld",
    ) -> None:
        """Initialize the VAE.

        Args:
            in_features: The number of input features.
            latent_dim: Dimension of the latent space.
            learning_rate: The learning rate for the optimizer.
            loss_type: The type of loss function to use ("mse_kld" or "elbo").
        """
        super().__init__()

        # Compute layer dimensions dynamically based on the number of input features.
        hidden_dim1 = in_features
        hidden_dim2 = in_features // 2
        hidden_dim3 = latent_dim * 2

        if latent_dim * 2 > hidden_dim2:
            raise ValueError(
                f"latent_dim * 2 ({latent_dim * 2}) cannot be larger than "
                f"hidden_dim2 ({hidden_dim2})."
            )

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.GELU(),
            nn.LayerNorm(hidden_dim1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim3),
        )

        self.mu_layer = nn.Linear(hidden_dim3, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim3, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim3),
            nn.GELU(),
            nn.LayerNorm(hidden_dim3),
            nn.Linear(hidden_dim3, hidden_dim2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim2),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, in_features),
        )

        self.learning_rate = learning_rate
        self.loss_type = loss_type

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input tensor into latent space.

        Args:
            x: The input tensor.

        Returns:
            A tuple of (mu, logvar) tensors.
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.

        Args:
            mu: The mean tensor.
            logvar: The log variance tensor.

        Returns:
            The sampled tensor.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent tensor back to input space.

        Args:
            z: The latent tensor.

        Returns:
            The reconstructed tensor.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE.

        Args:
            x: The input tensor.

        Returns:
            A tuple of (reconstructed tensor, mu, logvar).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def mse_kld_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the MSE + KLD loss.

        Args:
            recon_x: The reconstructed tensor.
            x: The input tensor.
            x_hat: The reconstructed tensor.
            mu: The mean tensor from the encoder.
            logvar: The log variance tensor from the encoder.

        Returns:
            A tuple of (total loss, reconstruction loss, KL divergence loss).
        """
        # Reconstruction loss.
        recon_loss = nn.MSELoss()(x_hat, x)

        # KL divergence loss.
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0) * x.size(1)  # Normalize by batch size and feature size.

        # Total loss.
        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss

    def elbo_loss(self, x, x_hat, mu, logvar):
        """Compute the ELBO loss with weighted reconstruction terms.

        Args:
            x: The input tensor.
            x_hat: The reconstructed tensor.
            mu: The mean tensor from the encoder.
            logvar: The log variance tensor from the encoder.

        Returns:
            A tuple of (total loss, reconstruction loss, KL divergence loss).
        """
        # Reconstruction loss with weighted MSE and L1 components.
        recon_mse = nn.functional.mse_loss(x_hat, x, reduction="none").sum(dim=1)
        recon_l1 = nn.functional.l1_loss(x_hat, x, reduction="none").sum(dim=1)
        recon_loss = (0.7 * recon_mse + 0.3 * recon_l1).mean()

        # KL divergence loss.
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        # Total loss.
        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step of the VAE.

        Args:
            batch: A tuple of (input tensor, target tensor).
            batch_idx: The batch index.

        Returns:
            The computed loss.
        """
        x, _ = batch
        recon_x, mu, logvar = self.forward(x)

        # Compute the loss based on the selected loss type.
        if self.loss_type == "mse_kld":
            total_loss, recon_loss, kl_loss = self.mse_kld_loss(x, recon_x, mu, logvar)
        elif self.loss_type == "elbo":
            total_loss, recon_loss, kl_loss = self.elbo_loss(x, recon_x, mu, logvar)
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

        # Logging.
        self.log(name="batch_idx", value=int(batch_idx), prog_bar=True)
        self.log(name="train_loss", value=total_loss, prog_bar=True)
        self.log(name="recon_loss", value=recon_loss, prog_bar=True)
        self.log(name="kl_loss", value=kl_loss, prog_bar=True)

        return total_loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test step of the VAE.

        Args:
            batch: A tuple of (input tensor, target tensor).
            batch_idx: The batch index.
        """
        x, _ = batch
        recon_x, mu, logvar = self.forward(x)

        # Compute the loss based on the selected loss type.
        if self.loss_type == "mse_kld":
            total_loss, recon_loss, kl_loss = self.mse_kld_loss(x, recon_x, mu, logvar)
        elif self.loss_type == "elbo":
            total_loss, recon_loss, kl_loss = self.elbo_loss(x, recon_x, mu, logvar)
        else:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")

        # Logging.
        self.log(name="test_loss", value=total_loss, prog_bar=True)
        self.log(name="test_recon_loss", value=recon_loss, prog_bar=True)
        self.log(name="test_kl_loss", value=kl_loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            The optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
