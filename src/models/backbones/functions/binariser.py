
from torch import nn


DEFAULT_THRESHOLD = 5e-3

class Binariser(nn.Module):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Binariser, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(self.threshold)] = 0
        outputs[inputs.gt(self.threshold)] = 1
        return outputs

    # def backward(self, gradOutput):
    #     return gradOutput
