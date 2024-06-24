from typing import List

import torch
from torch import nn

from models.backbones.functions import Binariser, Ternariser
from models.backbones.layers import WeightMaskedLinear


class WeightMaskedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 64,
    ):
        super().__init__()

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.activation = nn.ModuleList()

        self.layer_num = len(hidden_dims) + 1
        for l in range(self.layer_num):
            if l == 0:
                self.fc.append(WeightMaskedLinear(input_dim, hidden_dims[l]))
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))
            elif l == self.layer_num - 1:
                self.fc.append(WeightMaskedLinear(hidden_dims[l - 1], output_dim))
                self.bn.append(nn.BatchNorm1d(output_dim))
            else:
                self.fc.append(WeightMaskedLinear(hidden_dims[l - 1], hidden_dims[l]))
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))
            self.activation.append(nn.ReLU())

        for n, p in self.named_parameters():
            print(n, p)

    @property
    def mask(self):
        mask = {}
        for n, p in self.named_parameters():
            if "mask_real" in n:
                mask[n] = p
        return mask

    def forward(self, x, stage):

        for n, p in self.named_parameters():
            if n == "fc.0.mask_real":
                print(n, p)

        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        a = x.view(batch_size, -1)

        for l in range(self.layer_num):
            h = self.fc[l](a)
            h = self.bn[l](h)
            a = self.activation[l](h)

        return a


if __name__ == "__main__":
    _ = MLP()
