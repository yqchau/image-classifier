import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from utils.data_loader import get_data_loaders
from utils.model import PLModel


@hydra.main(version_base=None, config_path="./config", config_name="config")
def train(cfg: DictConfig):
    torch.manual_seed(cfg["train"]["seed"])

    train_data_loader, val_data_loader = get_data_loaders(**cfg["data_loader"])
    model = PLModel(**cfg["model"])
    trainer = pl.Trainer(**cfg["train"]["trainer"])

    trainer.fit(
        model=model,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )


if __name__ == "__main__":
    train()
