from torch import nn
import torch



class WeightProximityReg(nn.Module):
    """The weight proximity regularisation.

    See in EWC paper:
    https://arxiv.org/abs/
    """

    def __init__(self, factor: float):
        super().__init__()

        self.factor = factor  # regularisation factor

    def forward(self, backbone: nn.Module, previous_backbone: nn.Module, importance):
        reg = 0.0
        for (name,param),(_,param_old) in zip(backbone.named_parameters(),previous_backbone.named_parameters()):
            reg+=torch.sum(importance[name]*(param_old-param).pow(2))/2
        return (self.factor / 2) * reg
        