from typing import Any, Optional

import torch
from torch import nn
from lightning import LightningModule

from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()


class Finetuning(LightningModule):
    """LightningModule for naive finetuning continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self maintained task_id counter
        self.task_id: Optional[int] = None

        # store network module in self beyond self.hparams for convenience
        self.backbone = backbone
        self.heads = heads

        # loss function
        self.criterion = nn.CrossEntropyLoss()  # classification loss
        self.reg = None  # regularisation terms

    def forward(self, x: torch.Tensor, task_id: int):
        # the forward process propagates input to logits of classes of task_id
        feature = self.backbone(x)
        logits = self.heads(feature, task_id)
        return logits


    def _model_step(self, batch: Any, task_id: int):
        # common forward step among training, validation, testing step
        x, y = batch
        logits = self.forward(x, task_id)
        loss_cls = self.criterion(logits, y)
        loss_reg = 0.0
        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)

        return loss_cls, loss_reg, loss_total, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_reg, loss_total, preds, targets = self._model_step(
            batch, task_id=self.task_id
        )
        
        self.training_step_follow_up(loss_cls, loss_reg, loss_total, preds, targets)
        
        # return loss or backpropagation will fail
        return loss_total
        
    def training_step_follow_up(self, loss_cls, loss_reg, loss_total, preds, targets):

        # update metrics
        self.train_metrics[f"task{self.task_id}/train/loss/cls"](loss_cls)
        self.train_metrics[f"task{self.task_id}/train/loss/reg"](loss_reg)
        self.train_metrics[f"task{self.task_id}/train/loss/total"](loss_total)
        self.train_metrics[f"task{self.task_id}/train/acc"](preds, targets)

        # log_metrics
        loggerpack.log_train_metrics(self, self.train_metrics)



    def on_val_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_metrics[f"task{self.task_id}/val/loss/cls"].reset()
        self.val_metrics[f"task{self.task_id}/val/loss/reg"].reset()
        self.val_metrics[f"task{self.task_id}/val/loss/total"].reset()
        self.val_metrics[f"task{self.task_id}/val/acc"].reset()
        self.val_metrics[f"task{self.task_id}/val/acc/best"].reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss_cls, loss_reg, loss_total, preds, targets = self._model_step(
            batch, task_id=self.task_id
        )
        
        self.validation_step_follow_up(loss_cls, loss_reg, loss_total, preds, targets)

    def validation_step_follow_up(self, loss_cls, loss_reg, loss_total, preds, targets):

        # update metrics
        self.val_metrics[f"task{self.task_id}/val/loss/cls"](loss_cls)
        self.val_metrics[f"task{self.task_id}/val/loss/reg"](loss_reg)
        self.val_metrics[f"task{self.task_id}/val/loss/total"](loss_total)
        self.val_metrics[f"task{self.task_id}/val/acc"](preds, targets)

        # log metrics
        loggerpack.log_val_metrics(self, self.val_metrics)

    def on_validation_epoch_end(self):
        acc = self.val_metrics[
            f"task{self.task_id}/val/acc"
        ].compute()  # get current val acc
        self.val_metrics[f"task{self.task_id}/val/acc/best"](
            acc
        )  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            f"task{self.task_id}/val/acc_best",
            self.val_metrics[f"task{self.task_id}/val/acc/best"].compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss_cls, _, _, preds, targets = self._model_step(batch, dataloader_idx)

        self.test_step_follow_up(loss_cls, preds, targets, dataloader_idx, batch)
        
    def test_step_follow_up(self, loss_cls, preds, targets, dataloader_idx, batch):

        # update metrics
        self.test_metrics["test/loss/cls"][dataloader_idx](loss_cls)
        self.test_metrics["test/acc"][dataloader_idx](preds, targets)

        # log metrics
        loggerpack.log_test_metrics_progress_bar(
            self, self.test_metrics, dataloader_idx
        )

        loggerpack.log_test_samples(batch, preds, targets, dataloader_idx)

    def on_test_epoch_end(self):
        # update metrics
        print(self.task_id)
        for t in range(self.task_id + 1):
            self.test_metrics_overall[f"test/loss/cls/ave"](
                self.test_loss_cls[t].compute()
            )
            self.test_metrics_overall[f"test/acc/ave"](self.test_acc[t].compute())
            # self.test_metrics_overall[f"test/bwt"](self.test_acc[t].compute())

        # log metrics
        loggerpack.log_test_metrics(
            self, self.test_metrics, self.test_metrics_overall, task_id=self.task_id
        )

    def predict(self, batch: Any, task_id: int):
        """Pure prediction.
        
        Returns:
        preds (Tensor): predicted classes of batch.
        prods (Tensor): scores of predicted classes of batch.        
        """        
        logits = self.forward(batch, task_id)
        probs, preds = torch.max(logits, dim=1)        
        return preds, probs


    def configure_optimizers(self):
        # choose optimizers
        optimizer = self.hparams.optimizer(params=self.parameters())

        # choose learning-rate schedulers
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": f"task{self.task_id}/val/loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Finetuning(None, None, None, None)
