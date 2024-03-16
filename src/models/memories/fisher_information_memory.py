from .data_memory import DataMemory
from lightning import LightningModule

from copy import deepcopy

import torch
from torch import nn


class FisherInformationMemory:
    """Memory storing whole models (networks) for previous tasks.

    Args:
        s_max (float): max scale of mask gate function
    """

    def __init__(self, backbone: nn.Module):
        self.backbone = backbone

        # stores fisher information
        self.fis = {}

        # stores training data

    def empty_fi(self):
        fi = {}
        for n, p in self.backbone.named_parameters():
            fi[n] = 0 * p.data
        return fi

    def get_fi(self, task_id: int):
        """Get fisher information of task_id."""
        return self.fis[task_id]

    def calculate_fi(
        self, task_id: int, model: torch.nn.Module, criterion, training_data
    ):

        fi = self.empty_fi()

        N = 0
        for batch in training_data:
            x, y = batch
            num = x.size()[0]
            N += num
            # Forward and backward
            model.zero_grad()
            logits = model.forward(x, task_id)
            loss_cls = criterion(logits, y)
            loss_cls.backward()
            # Get gradients
            for n, p in model.backbone.named_parameters():
                if p.grad is not None:
                    fi[n] += num * p.grad.data.pow(2)

        # Mean
        for n, p in model.backbone.named_parameters():
            fi[n] /= N

        return fi

    def update(self, task_id: int, model: LightningModule, criterion, training_data):
        """Store fisher information of self.task_id after training it."""

        fi = self.calculate_fi(task_id, model, criterion, training_data)
        self.fis[task_id] = fi


if __name__ == "__main__":
    pass
