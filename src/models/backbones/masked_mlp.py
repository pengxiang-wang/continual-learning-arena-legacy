from typing import List

import torch
from torch import nn


class MaskedMLP(nn.Module):
    """Masked MLP for HAT algorithm."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 64,
    ):
        super().__init__()

        self.te = nn.ModuleDict()  # task embeddings over features
        self.mask_gate = nn.Sigmoid()
        self.test_mask = None

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.activation = nn.ModuleList()

        self.layer_num = len(hidden_dims) + 1
        for l in range(self.layer_num):
            if l == 0:
                self.fc.append(nn.Linear(input_dim, hidden_dims[l]))
                self.te[f"fc{l}"] = nn.Embedding(1, hidden_dims[l])
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))

            elif l == self.layer_num - 1:
                self.fc.append(nn.Linear(hidden_dims[l - 1], output_dim))
                self.te[f"fc{l}"] = nn.Embedding(1, output_dim)
                self.bn.append(nn.BatchNorm1d(output_dim))

            else:
                self.fc.append(nn.Linear(hidden_dims[l - 1], hidden_dims[l]))
                self.te[f"fc{l}"] = nn.Embedding(1, hidden_dims[l])
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))

            self.activation.append(nn.ReLU())

        self.test_mask = None

    def mask(self, task_embedding: nn.Embedding, scalar: float):
        return self.mask_gate(torch.tensor(scalar) * task_embedding.weight)

    def set_test_mask(self, mask):
        """Set task mask before testing task."""
        self.test_mask = mask
        # print(self.test_mask["fc1"])

    def forward(self, x, scalar: float, stage: str, additional_mask=None):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        a = x.view(batch_size, -1)

        mask_record = {}
        for l in range(self.layer_num):
            h = self.fc[l](a)
            m = (
                self.mask(self.te[f"fc{l}"], scalar)
                if stage == "fit"
                else self.test_mask[f"fc{l}"]
            )
            mask_record[f"fc{l}"] = m
            h = m * h  # apply mask

            if additional_mask:
                m_add = additional_mask[f"fc{l}"]
                h = m_add * h  # apply additional mask

            if stage == "train":  # problem! don't apply batchnorm at test stage
                h = self.bn[l](h)
            a = self.activation[l](h)

        # if stage == "test":
        # print(a)
        # print(self.mask(self.te[f"fc1"], scalar))
        # print(self.test_mask[f"fc1"])
        # print(a)
        return a, mask_record


if __name__ == "__main__":
    net = MaskedMLP(input_dim=784, hidden_dims=[256, 256], output_dim=64)
    print(net)
    for n, p in net.named_modules():
        print(n, p)
