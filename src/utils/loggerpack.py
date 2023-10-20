import csv
import os
from typing import Any, Callable, List

import numpy as np

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
import matplotlib.pyplot as plt


from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class LoggerPack:
    """Pack all kinds of loggers to be used everywhere.

    Logs in PyTorch Lightning only support scalars. We wrote this logger wrapper for other things to be logged.
    """

    def __init__(self, loggers: List[LightningLogger], log_dir: str):
        self.loggers = loggers  # Lightning logger
        self.log_dir = log_dir

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
            log.warning("Logger not found! Skipping hyperparameter logging...")
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
            
            
            
            
    def log_batch_predicts(self, task_id, imgs, preds, probs, targets = None):
        """Plot given images, along with predicted and true labels.

        """
        
        fig = plt.figure()
        fig.suptitle("Data from task {}".format(task_id))
        
        img_num = len(imgs)
        column = 4
        row = img_num // column + 1
        
        idx = 0
        for img in imgs:
            ax = fig.add_subplot(row, column, idx+1, xticks=[], yticks=[])
            plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            if targets:
                ax.set_title("pred: {0}, {1:.3f}\n(true label: {2})".format(
                    preds[idx],
                    probs[idx],
                    targets[idx]),
                            color=("green" if preds[idx]==targets[idx].item() else "red"))
            else:
                ax.set_title("predict: {0}, {1:.3f}".format(
                    preds[idx],
                    probs[idx]
                    ))

            idx += 1
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
            plt.imshow(m.detach(), aspect=10, cmap="Greys")
            plt.colorbar()

            mask_fig_path = os.path.join(
                mask_fig_dir, f"{module_name}_task{task_id}.png"
            )
            plt.savefig(mask_fig_path)

            if tensorboard:
                tensorboard.add_figure(f"test/mask/task{task_id}/{module_name}", fig)

        for module_name, m in previous_mask.items():
            fig = plt.figure()
            plt.imshow(m.detach(), aspect=10, cmap="Greys")
            plt.colorbar()

            mask_fig_path = os.path.join(
                mask_fig_dir, f"{module_name}_previous_task{task_id}.png"
            )
            plt.savefig(mask_fig_path)

            if tensorboard:
                tensorboard.add_figure(f"test/mask/previous/{module_name}", fig)

    # def check_lightning_module(log_func: Callable) -> Callable:
    #     """Decorator that checks if a log method in LoggerWrapper is executed within a LightningModule.

    #     If not, log a warning and continue.

    #     """
    #     def wrap()


def globalise_loggerpack(loggerpack_local: LoggerPack) -> LoggerPack:
    """Globalise loggerpack."""
    global loggerpack
    loggerpack = loggerpack_local


def get_global_loggerpack():
    """Get the globalised loggerpack."""
    return loggerpack
