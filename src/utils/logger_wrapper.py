import csv
import os
from typing import Any, Callable, List
import logging

import hydra
import numpy as np

from omegaconf import DictConfig
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
import matplotlib.pyplot as plt


class Logger:
    r"""
    A class that wraps anything about logging. It includes:
    - A Python logger object for command line logging (from Python built-in [logging](https://docs.python.org/3/library/logging.html) module). It can be called by `logger.pylogger`.
    - Lightning loggers objects such as [TensorBoardLogger](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#tensorboard) and [CSVLogger](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#csvlogger). They are passed as a list to Lightning Trainer object.
    - Other predefined logging fuctions using Python logger and Lightning loggers.

    Args:
        logger_cfg (DictConfig): Configuration composed by Hydra.
    """

    def __init__(
        self,
        logger_cfg: DictConfig,
        log_dir: str,
    ):
        self.pylogger = self.instantiate_pylogger(
            logger_cfg.pylogger
        )  # python logger object
        self.lightning_loggers = self.instantiate_lightning_loggers(
            logger_cfg.get("lightning_loggers")
        )  # Lightning loggers
        self.log_dir = log_dir  # some might need it

    def instantiate_pylogger(self, pylogger_logger_cfg: DictConfig) -> logging.Logger:
        r"""Initializes multi-GPU-friendly python command line logger."""

        pylogger = logging.getLogger(__name__)
        pylog_path = os.path.join(pylogger_logger_cfg.log_dir, pylogger_logger_cfg.filename)
        file_handler = logging.FileHandler(pylog_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt=pylogger_logger_cfg.fmt)
        file_handler.setFormatter(formatter)
        pylogger.addHandler(file_handler)

        # this ensures all logging levels get marked with the rank zero decorator
        # otherwise logs would get multiplied for each GPU process in multi-GPU setup
        logging_levels = (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
        )
        for level in logging_levels:
            setattr(pylogger, level, rank_zero_only(getattr(pylogger, level)))

        return pylogger

    def instantiate_lightning_loggers(
        self, lightning_logger_cfg: DictConfig
    ) -> List[LightningLogger]:
        """Instantiates loggers from config."""

        lightning_loggers: List[LightningLogger] = []

        if not lightning_logger_cfg:
            self.pylogger.warning("No lightning loggers config found! Skipping...")
            return lightning_loggers

        if not isinstance(lightning_logger_cfg, DictConfig):
            raise TypeError("Logger config must be a DictConfig!")

        for _, lg_conf in lightning_logger_cfg.items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                self.pylogger.info(f"Instantiating logger <{lg_conf._target_}>")
                lightning_loggers.append(hydra.utils.instantiate(lg_conf))

        return lightning_loggers

    @rank_zero_only
    def log_hyperparameters(self, object_dict: dict) -> None:
        """Controls which config parts are saved by lightning loggers.

        Additionally saves:
        - Number of model parameters
        """

        hparams = {}

        cfg = object_dict["cfg"]
        model = object_dict["model"]
        trainer = object_dict["trainer"]

        if not trainer.logger:
            logger.pylogger.warning(
                "Logger not found! Skipping hyperparameter logging..."
            )
            return

        hparams["model"] = cfg["model"]

        # save number of model parameters
        hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
        hparams["model/params/trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        hparams["model/params/non_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

        hparams["data"] = cfg["data"]
        hparams["trainer"] = cfg["trainer"]

        hparams["callbacks"] = cfg.get("callbacks")
        hparams["extras"] = cfg.get("extras")

        hparams["experiment_name"] = cfg.get("experiment_name")
        hparams["tags"] = cfg.get("tags")
        hparams["ckpt_path"] = cfg.get("ckpt_path")
        hparams["seed"] = cfg.get("seed")

        # send hparams to all loggers
        for logger in self.loggers:
            logger.log_hyperparams(hparams)

    def log_train_metrics(self, model: LightningModule, train_metrics: dict) -> None:
        """Log train metrics to loggers and progress bar.

        You should only execute it within LightningModule and provide it.
        """
        for metric_name, metric in train_metrics.items():
            model.log(
                metric_name,
                metric,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

    def log_val_metrics(self, model: LightningModule, val_metrics: dict) -> None:
        """Log train metrics to loggers and progress bar.

        You should only execute it within LightningModule and provide it.
        """
        for metric_name, metric in val_metrics.items():
            model.log(
                metric_name,
                metric,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

    def log_test_metrics_progress_bar(
        self, model: LightningModule, test_metrics: dict, dataloader_idx: int
    ) -> None:
        """Log train metrics to loggers and progress bar.

        You should only execute it within LightningModule and provide the log method in lightning_log_func arg.
        """

        # don't log to loggers, only progress bar
        for metric_name, metric in test_metrics.items():
            model.log(
                metric_name,
                metric[dataloader_idx],
                add_dataloader_idx=True,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

    def log_test_samples(
        self, batch: Any, preds: torch.Tensor, targets: torch.Tensor, task_id_test: int
    ) -> None:
        """Log a small proportion of test samples to logger."""

        for logger in self.loggers:
            if type(logger) == TensorBoardLogger:
                tensorboard = logger.experiment
                pass

    def log_test_metrics(
        self,
        model: LightningModule,
        test_metrics: dict,
        test_metrics_overall: dict,
        task_id: int,
    ) -> None:
        to_be_writed = {"task": task_id}
        for test_metric_name, test_metric in test_metrics_overall.items():
            to_be_writed[test_metric_name] = test_metric.compute().item()
        for task_id_test in range(task_id + 1):
            for test_metric_name, test_metric in test_metrics.items():
                to_be_writed[f"{test_metric_name}/task{task_id_test}"] = (
                    test_metric[task_id_test].compute().item()
                )

        # log to loggers
        for test_metric_name, test_metric in test_metrics_overall.items():
            model.log(
                test_metric_name,
                test_metric,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        # log to csv file
        # CSVLogger in Lightning can't be customised to store test metrics. Instead we use Python csv module.
        csv_dir = os.path.join(self.log_dir, "csv/")
        csv_path = os.path.join(csv_dir, "test_metrics.csv")
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)

        first = not os.path.exists(csv_path)
        if not first:
            with open(csv_path, "r") as file:
                lines = file.readlines()
                del lines[0]
        # write header
        with open(csv_path, "w") as file:
            writer = csv.DictWriter(file, fieldnames=to_be_writed.keys())
            writer.writeheader()
        # write metrics
        with open(csv_path, "a") as file:
            if not first:
                file.writelines(lines)
            writer = csv.DictWriter(file, fieldnames=to_be_writed.keys())
            writer.writerow(to_be_writed)

    def log_predicts(self, task_id, imgs, preds, probs, targets=None, attrs=None):
        """Plot given images, along with predicted and true labels, attrs generated by interpreters."""

        fig = plt.figure()
        fig.suptitle("Data from task {}".format(task_id))

        img_num = len(imgs)
        column = 4
        row = (2 * img_num) // column + 1

        for idx, img in enumerate(imgs):

            # plot figure
            ax = fig.add_subplot(row, column, 2 * idx + 1, xticks=[], yticks=[])
            plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            if targets != None:
                ax.set_title(
                    "pred: {0}, {1:.3f}\n(true label: {2})".format(
                        preds[idx], probs[idx], targets[idx]
                    ),
                    color=("green" if preds[idx] == targets[idx].item() else "red"),
                )
            else:
                ax.set_title("predict: {0}, {1:.3f}".format(preds[idx], probs[idx]))

            # plot attribution
            if attrs != None:
                ax = fig.add_subplot(row, column, 2 * idx + 2, xticks=[], yticks=[])
                plt.imshow(np.transpose(attrs[idx].numpy(), (1, 2, 0)))
                ax.set_title("attr")

        plt.show()

        return fig

    def log_train_mask(
        self,
        mask,
        task_id: int,
        step: int,
        every_n_train_steps: int = 50,
        plot_figure: bool = False,
    ):
        if step % every_n_train_steps != 0:
            return
        mask_src_dir = os.path.join(self.log_dir, "mask/", "train/", "src/")
        if not os.path.exists(mask_src_dir):
            try:
                os.mkdir(mask_src_dir)
            except:
                os.makedirs(mask_src_dir)

        mask_src_path = os.path.join(mask_src_dir, f"task{task_id}_step{step}.pt")
        torch.save(mask, mask_src_path)

        if plot_figure:
            for module_name, m in mask.items():
                fig = plt.figure()
                plt.imshow(m.detach(), aspect=10, cmap="Greys")
                plt.colorbar()

                mask_fig_dir = os.path.join(self.log_dir, "mask/", "train/", "fig/")
                if not os.path.exists(mask_fig_dir):
                    try:
                        os.mkdir(mask_fig_dir)
                    except:
                        os.makedirs(mask_fig_dir)

                mask_fig_path = os.path.join(
                    mask_fig_dir, f"{module_name}_task{task_id}_step{step}.png"
                )
                plt.savefig(mask_fig_path)

    def log_test_mask(
        self,
        mask,
        previous_mask,
        task_id: int,
    ):
        mask_src_dir = os.path.join(self.log_dir, "mask/", "test/", "src/")
        if not os.path.exists(mask_src_dir):
            try:
                os.mkdir(mask_src_dir)
            except:
                os.makedirs(mask_src_dir)

        mask_src_path = os.path.join(mask_src_dir, f"task{task_id}.pt")
        torch.save(mask, mask_src_path)

        mask_fig_dir = os.path.join(self.log_dir, "mask/", "test/", "fig/")
        if not os.path.exists(mask_fig_dir):
            try:
                os.mkdir(mask_fig_dir)
            except:
                os.makedirs(mask_fig_dir)

        tensorboard = None
        for logger in self.loggers:
            if type(logger) == TensorBoardLogger:
                tensorboard = logger.experiment  # show mask fig in tensorboard

        for module_name, m in mask.items():
            fig = plt.figure()
            plt.imshow(m.detach().cpu(), aspect=10, cmap="Greys")
            plt.colorbar()

            mask_fig_path = os.path.join(
                mask_fig_dir, f"{module_name}_task{task_id}.png"
            )
            plt.savefig(mask_fig_path)

            if tensorboard:
                tensorboard.add_figure(f"test/mask/task{task_id}/{module_name}", fig)

        for module_name, m in previous_mask.items():
            fig = plt.figure()
            plt.imshow(m.detach().cpu(), aspect=10, cmap="Greys")
            plt.colorbar()

            mask_fig_path = os.path.join(
                mask_fig_dir, f"{module_name}_previous_task{task_id}.png"
            )
            plt.savefig(mask_fig_path)

            if tensorboard:
                tensorboard.add_figure(f"test/mask/previous/{module_name}", fig)

    def log_capacity(
        self,
        capacity,
        task_id: int,
        step: int,
    ):
        capacity = float(capacity)
        capacity_path = os.path.join(self.log_dir, "capacity.csv")
        if not os.path.exists(capacity_path):
            with open(capacity_path, "w", newline="") as csvfile:
                pass

        with open(capacity_path, "a+", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([capacity])

    # def check_lightning_module(log_func: Callable) -> Callable:
    #     """Decorator that checks if a log method in LoggerWrapper is executed within a LightningModule.

    #     If not, log a warning and continue.

    #     """
    #     def wrap()

    # loggers all in one pack
    # make logger available across all modules. There must be only one logger instance.
    # after globalising, use `utils.get_global_logger()` in other modules.

logger = None
def set_logger_global(logger_local: Logger):
    """Set the globalised logger."""
    global logger
    logger = logger_local

def get_logger():
    """Get the globalised logger."""
    return logger
