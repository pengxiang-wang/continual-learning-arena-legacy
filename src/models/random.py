from typing import Any, Optional

import torch
from torch import nn
from lightning import LightningModule


from src.models.memories import DataMemory
from src.utils import pylogger, loggerpack

log = pylogger.get_pylogger(__name__)
loggerpack = loggerpack.get_global_loggerpack()


class Random(LightningModule):
    """LightningModule for continual learning with random."""

    def __init__(self):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self maintained task_id counter
        self.task_id: Optional[int] = None

        self.training_data_memory = DataMemory()



    def training_step(self, batch: Any, batch_idx: int):

        # save 
        self.training_data_memory.update(batch, self.task_id) # for calculating fisher information
        
        loss_total = 0.0

        # return loss or backpropagation will fail
        return loss_total

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        loss_cls, _, _, preds, targets = self._model_step(batch, dataloader_idx)

        if dataloader_idx == 1:
            pass
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
