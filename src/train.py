import logging

import hydra

from typing import List

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import log_hyperparameters

log = logging.getLogger(__name__)


def train(config: DictConfig) -> None:
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info(f"Instantiating algorithm {config.algorithm._target_}")
    algorithm: LightningModule = hydra.utils.instantiate(
        config.algorithm,
        network=None,  # instead, we give network_conf
        network_conf=(OmegaConf.to_yaml(config.network) if "network" in config else None),
        optimizer_conf=(OmegaConf.to_yaml(config.optimizer) if "optimizer" in config else None),
        scheduler_conf=(OmegaConf.to_yaml(config.scheduler) if "scheduler" in config else None)
    )
    
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    loggers: List[LightningLoggerBase] = []
    if "loggers" in config:
        for name, lg_conf in config.loggers.items():
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = hydra.utils.instantiate(lg_conf)
            loggers.append(logger)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    log_hyperparameters(config=config, algorithm=algorithm, trainer=trainer)

    # Train the model
    log.info("Starting training!")
    trainer.fit(algorithm, datamodule=datamodule)

    trainer.test(dataloaders=datamodule.test_dataloader())
