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
        self.num = 0

        # stores training data

    def empty_fi(self):
        fi = {}
        for n, p in self.backbone.named_parameters():
            fi[n] = 0 * p.data
        return fi

    def get_fi(self, task_id: int):
        """Get fisher information of task_id."""
        return self.fis[task_id]
    
    # def calculate_fi(
    #     self, task_id: int, model: torch.nn.Module, criterion, training_data
    # ):

    #     fi = self.empty_fi()

    #     model.train()

    #     N = 0
    #     for batch in training_data:
    #         x, y = batch
    #         num = x.size()[0]
    #         N += num
    #         # Forward and backward
    #         model.zero_grad()
    #         logits = model.forward(x, task_id)
    #         loss_cls = criterion(logits, y)
    #         loss_cls.backward()
    #         # Get gradients
    #         for n, p in model.backbone.named_parameters():
    #             if p.grad is not None:
    #                 fi[n] += num * p.grad.data.pow(2)
    #                 print(fi[n].size())

    #     # Mean
    #     for n, p in model.backbone.named_parameters():
    #         fi[n] /= N

    #     return fi

    def update(self, task_id: int, model_grad_computed, batch_size):
        """Updata fisher information of self.task_id during training."""
        if task_id not in self.fis.keys():
            print(task_id,self.fis.keys() )
            self.fis[task_id] = self.empty_fi()
            self.num = 0
        # Get gradients
        for n, p in model_grad_computed.backbone.named_parameters():
            # if task_id == 1: print(p)
            # if p.grad is not None:
            self.fis[task_id][n] += batch_size * p.grad.data.pow(2)
            self.num += batch_size
            # print(self.num)
        

    def calculate_mean(self, task_id):
        for n, p in self.backbone.named_parameters():
            print(self.num)

            self.fis[task_id][n] /= self.num


if __name__ == "__main__":
    pass
