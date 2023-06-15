from typing import List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: List[int] = [256, 256],
        output_size: int = 64,
    ):
        super().__init__()

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.activation = nn.ModuleList()

        self.layer_num = len(hidden_size) + 1
        for l in range(self.layer_num):
            if l == 0:
                self.fc.append(nn.Linear(input_size, hidden_size[l]))
                self.bn.append(nn.BatchNorm1d(hidden_size[l]))
            elif l == self.layer_num - 1:
                self.fc.append(nn.Linear(hidden_size[l - 1], output_size))
                self.bn.append(nn.BatchNorm1d(output_size))
            else:
                self.fc.append(nn.Linear(hidden_size[l - 1], hidden_size[l]))
                self.bn.append(nn.BatchNorm1d(hidden_size[l]))
            self.activation.append(nn.ReLU())

    def forward(self, x):
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
