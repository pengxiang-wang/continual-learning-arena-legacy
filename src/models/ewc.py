from typing import Any

from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import pyrootutils
import torch
from torch import nn

pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.models import Finetuning
from src.models.memories import FisherInformationMemory, DataMemory, ModelMemory
from src.utils import logger, logger

log = logger.get_pylogger(__name__)
logger = logger.get_global_logger()


class EWC(Finetuning):
    """LightningModule for EWC () continual learning algorithm."""

    def __init__(
        self,
        heads: torch.nn.Module,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        reg: torch.nn.Module,
    ):
        super().__init__(heads, backbone, optimizer, scheduler)

        # LwF regularisation loss function
        self.reg = reg

        # Memory store fisher information of each task
        self.fisher_information_memory = FisherInformationMemory(backbone=backbone)
        # self.training_data_memory = DataMemory()
        self.model_memory = ModelMemory()

        # manual optimization
        self.automatic_optimization = False

    def on_train_end(self):
        self.model_memory.update(backbone=self.backbone, heads=self.heads)
        self.fisher_information_memory.calculate_mean(task_id=self.task_id)
        # training_data = self.training_data_memory.get_data()
        # self.training_data_memory.release()

    def training_step(self, batch: Any, batch_idx: int):
        # common forward step among training, validation, testing step
        x, y = batch

        # self.training_data_memory.update(batch, self.task_id) # for calculating fisher information

        logits = self.forward(x, self.task_id)
        loss_cls = self.criterion(logits, y)
        loss_cls.backward(retain_graph=True)
        self.fisher_information_memory.update(
            task_id=self.task_id, model_grad_computed=self, batch_size=len(batch)
        )

        opt = self.optimizers()
        opt.zero_grad()

        loss_reg = 0.0
        previous_backbone = self.model_memory.get_backbone()
        for previous_task_id in range(self.task_id):
            fi = self.fisher_information_memory.get_fi(previous_task_id)
            loss_reg += self.reg(self.backbone, previous_backbone, fi)

        loss_total = loss_cls + loss_reg
        preds = torch.argmax(logits, dim=1)

        # backward step
        self.manual_backward(loss_total)
        opt.step()

        self.training_step_follow_up(loss_cls, loss_reg, loss_total, preds, y)

        return loss_total


if __name__ == "__main__":
    _ = EWC(None, None, None, None, None)
