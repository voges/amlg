"""MLOps playground revolving around an autoencoder trained on the MNIST dataset."""

import logging

import hydra
import lightning as l
import optuna
from data import create_data_loaders, load_data
from models import AutoEncoder
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def train_and_test_model(cfg: DictConfig, latent_dim: int) -> float:
    """Train and test the autoencoder model."""
    # Load the data and create data loaders.
    train_set, test_set = load_data()
    train_loader, test_loader = create_data_loaders(
        cfg=cfg, train_set=train_set, test_set=test_set
    )

    # Create an instance of the autoencoder model.
    autoencoder = AutoEncoder(lr=cfg.train.learning_rate, latent_dim=latent_dim)

    # Create a trainer.
    trainer = l.Trainer(max_epochs=cfg.train.max_epochs)

    # Train the autoencoder.
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

    # Test the model.
    trainer.test(model=autoencoder, dataloaders=test_loader)
    test_loss = trainer.callback_metrics["test_loss"].item()

    return test_loss


class Objective:
    """Objective function for the hyperparameter optimization."""

    def __init__(self, cfg: DictConfig) -> None:  # noqa: D107
        self.cfg = cfg

    def __call__(self, trial: optuna.Trial) -> float:  # noqa: D102
        optuna_study_name = trial.study.study_name
        optuna_trial_number = trial.number

        log.info(f"Optuna study name: {optuna_study_name}")
        log.info(f"Optuna trial number: {optuna_trial_number}")

        latent_dim = trial.suggest_int("latent_dim", low=2, high=8)

        test_loss = train_and_test_model(cfg=self.cfg, latent_dim=latent_dim)

        return test_loss


def log_yaml(yaml_dump: str) -> None:
    """Pretty log of a YAML dump.

    Args:
        yaml_dump: The YAML dump to log.
    """
    for line in yaml_dump.splitlines():
        log.info(f"{line}")
    log.info("")


def log_config(cfg: DictConfig) -> None:
    """Pretty log of a Hydra configuration.

    Args:
        cfg: The configuration object.
    """
    log.info("")
    log.info("Configuration:")
    log.info("--------------")
    log_yaml(yaml_dump=OmegaConf.to_yaml(cfg=cfg))


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Optimize hyperparameters of the autoencoder model."""
    try:
        log_config(cfg=cfg)

        study = optuna.create_study(
            study_name=cfg.optuna.study_name,
            direction="minimize",
            storage=cfg.optuna.db_path,
            load_if_exists=True,
        )

        study.optimize(func=Objective(cfg=cfg), n_trials=cfg.optuna.n_trials)

        log.info(f"Best parameters: {study.best_trial.params}")
    except Exception as e:
        log.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
