from typing import List, Tuple

import hydra
import pyrootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger as LightningLogger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src from src import utils
from callbacks import ContinualCheckpoint
from src.utils import Logger

# prepare loggers
log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint of continual learning model on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path  # checkpoint path must be provided

    # prepare loggers
    logger.pylogger.info("Instantiating loggers...")
    lightning_loggers: List[LightningLogger] = utils.instantiate_lightning_loggers(
        cfg.get("logger")
    )

    logger: Logger = Logger(
        loggers=lightning_loggers, log_dir=cfg.paths.output_dir
    )  # loggers all in one pack
    # make logger available across all modules. There must be only one logger instance.
    # after globalising, use `utils.get_global_logger()` in other modules.
    utils.globalise_logger(logger)

    logger.pylogger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.pylogger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.pylogger.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    callbacks.extend([ContinualCheckpoint()])

    # trainer for evaluation
    logger.pylogger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=lightning_loggers
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": lightning_loggers,
        "trainer": trainer,
    }

    if lightning_loggers:
        logger.pylogger.info("Logging hyperparameters!")
        logger.log_hyperparameters(object_dict=object_dict)

    utils.continual_utils.set_test(
        model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path
    )

    logger.pylogger.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
