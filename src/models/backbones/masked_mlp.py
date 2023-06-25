from typing import List

import torch
from torch import nn


class MaskedMLP(nn.Module):
    """Masked MLP for HAT algorithm."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: List[int] = [256, 256],
        output_size: int = 64,
    ):
        super().__init__()

        self.te = nn.ModuleDict()  # task embeddings over features
        self.mask_gate = nn.Sigmoid()

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.activation = nn.ModuleList()

        self.layer_num = len(hidden_size) + 1
        for l in range(self.layer_num):
            if l == 0:
                self.fc.append(nn.Linear(input_size, hidden_size[l]))
                self.te[f"fc{l}"] = nn.Embedding(1, hidden_size[l])
                self.bn.append(nn.BatchNorm1d(hidden_size[l]))

            elif l == self.layer_num - 1:
                self.fc.append(nn.Linear(hidden_size[l - 1], output_size))
                self.te[f"fc{l}"] = nn.Embedding(1, output_size)
                self.bn.append(nn.BatchNorm1d(output_size))

            else:
                self.fc.append(nn.Linear(hidden_size[l - 1], hidden_size[l]))
                self.te[f"fc{l}"] = nn.Embedding(1, hidden_size[l])
                self.bn.append(nn.BatchNorm1d(hidden_size[l]))

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

        mask = {}
        for l in range(self.layer_num):
            h = self.fc[l](a)
            m = (
                self.mask(self.te[f"fc{l}"], scalar)
                if stage == "fit"
                else self.test_mask[f"fc{l}"]
            )
            h = m * h  # apply mask
            if additional_mask:
                m_add = additional_mask[f"fc{l}"]
                h = m_add * h  # apply additional mask
            if stage == "train":  # problem! don't apply batchnorm at test stage
                h = self.bn[l](h)
            a = self.activation[l](h)

            mask[f"fc{l}"] = m

        # if stage == "test":
        # print(a)
        # print(self.mask(self.te[f"fc1"], scalar))
        # print(self.test_mask[f"fc1"])
        # print(a)
        return a, mask


if __name__ == "__main__":
    net = MaskedMLP(input_size=784, hidden_size=[256, 256], output_size=64)
    print(net)
    for n, p in net.named_modules():
        print(n, p)
