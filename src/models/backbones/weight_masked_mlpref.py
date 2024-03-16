from typing import List

import torch
from torch import nn


class WeightMaskedMLP(nn.Module):
    """Masked MLP for HAT algorithm."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 64,
    ):
        super().__init__()

        self.wm = nn.ModuleDict()  # weight masks over parameters (weights)
        # self.mask_gate = nn.??()
        self.test_mask = None

        self.fc = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.activation = nn.ModuleList()

        self.layer_num = len(hidden_dims) + 1
        for l in range(self.layer_num):
            if l == 0:
                self.fc.append(nn.Linear(input_dim, hidden_dims[l]))
                self.wm[f"fc{l}"] = nn.Parameter(
                    torch.zeros(input_dim, hidden_dims[l], requires_grad=True)
                )
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))

            elif l == self.layer_num - 1:
                self.fc.append(nn.Linear(hidden_dims[l - 1], output_dim))
                self.wm[f"fc{l}"] = nn.Parameter(
                    torch.zeros(hidden_dims[l - 1], output_dim, requires_grad=True)
                )

                self.bn.append(nn.BatchNorm1d(output_dim))

            else:
                self.fc.append(nn.Linear(hidden_dims[l - 1], hidden_dims[l]))

                self.wm[f"fc{l}"] = nn.Parameter(
                    torch.zeros(hidden_dims[l - 1], hidden_dims[l], requires_grad=True)
                )
                self.bn.append(nn.BatchNorm1d(hidden_dims[l]))

            self.activation.append(nn.ReLU())

        self.test_mask = None

        # alternative

        for n, p in self.named_parameters():
            mask[n] = 0 * p.data

    def mask(self, task_embedding: nn.Embedding, scalar: float):
        return self.mask_gate(torch.tensor(scalar) * task_embedding.weight)

    def set_test_mask(self, mask):
        """Set task mask before testing task."""
        self.test_mask = mask
        # print(self.test_mask["fc1"])

    # def register_mask_hook(self):
    #     h = {}
    #     for n,p in self.named_parameters():
    #         h[n] = p.register_hook(lambda grad: grad * self.wm[n])

    #     return h

    # def remove_mask_hook(self, h):
    #     for n,p in self.named_parameters():
    #         h.remove()

    def forward(self, x, stage: str):
        batch_size, channels, width, height = x.size()
        orig_params = self.parameters()

        for (name, param), (_, param_old) in zip(
            backbone.named_parameters(), previous_backbone.named_parameters()
        ):
            reg += torch.sum(importance[name] * (param_old - param).pow(2)) / 2

        # (batch, 1, width, height) -> (batch, 1*width*height)
        a = x.view(batch_size, -1)

        for l in range(self.layer_num):

            def hook_fn(grad):
                print(g)
                g = 2 * grad
                return g

            z.register_hook(hook_fn)

            masked_weights = self.fc[l].weight.data * self.wm[f"fc{l}"]
            # apply mask
            h = self.fc[l](a)
            m = (
                self.mask(self.wm[f"fc{l}"], scalar)
                if stage == "fit"
                else self.test_mask[f"fc{l}"]
            )

            if stage == "train":  # problem! don't apply batchnorm at test stage
                h = self.bn[l](h)
            a = self.activation[l](h)

        # if stage == "test":
        # print(a)
        # print(self.mask(self.te[f"fc1"], scalar))
        # print(self.test_mask[f"fc1"])
        # print(a)
        return a


if __name__ == "__main__":
    net = MaskedMLP(input_dim=784, hidden_dims=[256, 256], output_dim=64)
    print(net)
    for n, p in net.named_modules():
        print(n, p)
