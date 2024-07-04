import os
from typing import List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig

import pyrootutils

# setup root directory for the project, so that all paths are relative to the project root
pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

# import our own modules
# because of the setup_root in train.py and so on, we can import from src without any problems
from src import utils
from src.callbacks import ContinualCheckpoint, ContinualProgressBar
from src.utils import Logger, set_logger_global


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the continual learning model. Can additionally evaluate on a testset (using best weights obtained during
    training).

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # Instantiate loggers
    logger: Logger = Logger(logger_cfg=cfg.logger, log_dir=cfg.paths.log_dir)
    set_logger_global(logger)
    logger.pylogger.debug(f"Logger initiatiated!")

    # Set global seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        logger.pylogger.debug(f"Global seed is set as {cfg.seed}!")

    # Instantiate Lightning Datamodule
    logger.pylogger.debug(f"Instantiating Lightning Datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    logger.pylogger.debug(f"Lightning Datamodule <{cfg.data._target_}> instantiated!")

    # Instantiate Lightning Module
    logger.pylogger.debug(f"Instantiating Lightning Module <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    logger.pylogger.debug(f"Lightning Module <{cfg.model._target_}> instantiated!")

    # Compile model
    if cfg.get("compile"):
        model = torch.compile(model)
    logger.pylogger.info("Model is compiled by PyTorch!")


    logger.pylogger.info("Start training!")
    for task_id in range(datamodule.num_tasks):
        logger.pylogger.info("Instantiating callbacks...")
        callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
        callbacks.extend([ContinualCheckpoint(), ContinualProgressBar()])

        profiler = SimpleProfiler(
            dirpath=os.path.join(cfg.paths.output_dir, "profilers"),
            filename=f"task{task_id}",
        )

        logger.pylogger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            profiler=profiler,
        )

        # torch.cuda.set_device(trainer.accelerator.devices)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            logger.pylogger.info("Logging hyperparameters!")
            logger.log_hyperparameters(object_dict=object_dict)

        utils.continual_utils.set_task_train(
            task_id=task_id,
            datamodule=datamodule,
            model=model,
            trainer=trainer,
        )

        logger.pylogger.info(f"Starting training task {task_id}!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        train_metrics = trainer.callback_metrics

        if cfg.get("test"):
            logger.pylogger.info(
                f"Starting testing model after training task {task_id}!"
            )
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                logger.pylogger.warning(
                    "Best ckpt not found! Using current weights for testing..."
                )
                ckpt_path = None
            logger.pylogger.info(f"Best ckpt path: {ckpt_path}")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
