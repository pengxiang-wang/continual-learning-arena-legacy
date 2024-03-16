from typing import List
from copy import deepcopy


import torch
from torch import nn


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
                self.fc.append(nn.Linear(input_dim, hidden_dims[l]))
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))
            elif l == self.layer_num - 1:
                self.fc.append(nn.Linear(hidden_dims[l - 1], output_dim))
                self.bn.append(nn.BatchNorm1d(output_dim))
            else:
                self.fc.append(nn.Linear(hidden_dims[l - 1], hidden_dims[l]))
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))
            self.activation.append(nn.ReLU())

        self.params = {}

        self.mask = {}
        for n, p in self.named_parameters():
            self.mask[n] = torch.ones_like(p.data)
            # print(type(self.mask[n]))

        self.test_mask = None

    def set_test_mask(self, mask):
        """Set task mask before testing task."""
        self.test_mask = mask

    def forward(self, x, stage):
        batch_size, channels, width, height = x.size()

        for n, p in self.named_parameters():
            print(type(p))
            # self.params[n] = deepco
            # py(p).requires_grad_()
            if n == "fc.0.weight":
                print("p.data", p.data)
            # print(p.data)
            p = self.params[n] * (self.mask[n] if stage == "fit" else self.test_mask[n])
            # print(p.data)
        print("param", self.params["fc.0.weight"])
        # print(self.mask["fc.0.weight"])

        # (batch, 1, width, height) -> (batch, 1*width*height)
        a = x.view(batch_size, -1)

        for l in range(self.layer_num):
            h = self.fc[l](a)
            h = self.bn[l](h)
            a = self.activation[l](h)

        return a

    def take_off_mask(self):
        for n, p in self.named_parameters():
            p.data = self.params[n]


if __name__ == "__main__":
    _ = WeightMaskedMLP()
