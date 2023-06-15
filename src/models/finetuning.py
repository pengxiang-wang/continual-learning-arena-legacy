from typing import Any, Dict, Optional

import torch
from torch import nn
from lightning import LightningModule


class Finetuning(LightningModule):
    """LightningModule for naive finetuning continual learning algorithm."""

    def __init__(
        self,
        head: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.task_id = 0

        self.backbone = backbone
        self.head = head

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, task_id: int):
        # the forward process propagates input to logits of classes of task_id
        feature = self.backbone(x)
        logits = self.head(feature, task_id)
        return logits

    def model_step(self, batch: Any, task_id: int):
        x, y = batch
        logits = self.forward(x, task_id)
        loss_cls = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss_cls, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss_cls, preds, targets = self.model_step(batch, self.task_id)

        # update and log metrics
        self.train_loss_cls(loss_cls)
        self.train_acc(preds, targets)
        self.log(
            f"train/loss/cls",
            self.train_loss_cls,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss_cls

    def on_val_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_cls.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss_cls, preds, targets = self.model_step(batch, self.task_id)

        # update and log metrics
        self.val_loss_cls(loss_cls)
        self.val_acc(preds, targets)
        self.log(
            f"val/loss/cls",
            self.val_loss_cls,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            f"val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss_cls, preds, targets = self.model_step(batch, dataloader_idx)
        # update and log metrics
        self.test_loss_cls[dataloader_idx](loss_cls)
        self.test_acc[dataloader_idx](preds, targets)
        self.log(
            f"test/loss/cls",
            self.test_loss_cls[dataloader_idx],
            add_dataloader_idx=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"test/acc",
            self.test_acc[dataloader_idx],
            add_dataloader_idx=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        # test all seen task till self.task_id
        for t in range(self.task_id + 1):
            self.ave_test_loss_cls(self.test_loss_cls[t].compute())
            self.ave_test_acc(self.test_acc[t].compute())
        self.log(f"test/loss/cls/ave", self.ave_test_loss_cls)
        self.log(f"test/acc/ave", self.ave_test_acc)

    def configure_optimizers(self):
        # Choose what optimizers and learning-rate schedulers to use in your optimization
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss/cls",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Finetuning(None, None, None, None)
