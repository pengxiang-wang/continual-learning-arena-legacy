from typing import List, Tuple

import hydra
import pyrootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger as LightningLogger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
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

from src import utils
from src.callbacks import ContinualCheckpoint
from src.utils import LoggerPack

# prepare loggers
log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """Predicts batches from given dataloader with given checkpoint of continual learning model. It serves in a interactive way.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path # checkpoint path must be provided
    
    # prepare loggers
    log.info("Instantiating loggers...")
    lightning_loggers: List[LightningLogger] = utils.instantiate_lightning_loggers(
        cfg.get("logger")
    )
    
    loggerpack: LoggerPack = LoggerPack(
        loggers=lightning_loggers, log_dir=cfg.paths.output_dir
    ) # loggers all in one pack
    # make loggerpack available across all modules. There must be only one loggerpack instance. 
    # after globalising, use `utils.get_global_loggerpack()` in other modules.
    utils.globalise_loggerpack(loggerpack)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)


    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg.get("callbacks")
    )
    callbacks.extend([ContinualCheckpoint()])
    
    # trainer for evaluation
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
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
        log.info("Logging hyperparameters!")
        loggerpack.log_hyperparameters(object_dict=object_dict)
        
    utils.continual_utils.set_predict(
        model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path
    )
    
    log.info("Ready for predicting!")
    input_source = input("Where are your inputs from? \n1. test dataset (datamodule); \n2. external images; \nq. quit: \n")
    while input_source != "q":
        
        # get ready for data and labels
        task_id = int(input("Which task is your input from? Task ID: "))
        if input_source == "1":
            test_set_t = datamodule.data_test_orig[task_id] # get test data of task ID
            img_idx = [int(idx) for idx in input("Which images? Index (allow multiple): ").split(" ")] # input image indices
            imgs, batch, targets = utils.read_dataset_images(test_set_t, img_idx=img_idx, normalize_transform=datamodule.normalize_transform)

        elif input_source == "2":
            input_img_path = input("Your image path (allow multiple) or directory: ")
            imgs, batch = utils.read_image(input_img_path, normalize_transform=datamodule.normalize_transform)
            targets = input("Your image true label: ").split(" ")
            
        # predicting
        preds, probs = model.predict(batch, task_id)
        # visualisation
        loggerpack.log_batch_predicts(task_id, imgs, preds, probs, targets=targets)
        
        input_source = input("Where are your inputs from? \n1. test dataset (datamodule); \n2. external images; \nq. quit: \n")
        
        
    
    
    
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    predict(cfg)


if __name__ == "__main__":
    main()
