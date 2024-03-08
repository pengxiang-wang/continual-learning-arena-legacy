from typing import List

import torch
from torch import nn
from torch.autograd import Variable

from src.models.backbones.functions import Binariser, Ternariser
from torch.nn.parameter import Parameter

import torch.nn.functional as F

import math

DEFAULT_THRESHOLD = 5e-3



class WeightMaskedLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(WeightMaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
        # self.weight = Parameter(torch.empty((out_features, in_features)))
        # if bias:
        #     self.bias = Parameter(torch.empty(out_features))
        # else:
        #     self.register_parameter('bias', None)
            

        self.weight = Variable(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.weight.data.normal_(0, 1)
        self.bias.data.normal_(0, 0.001)


        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        if mask_init == '1s':
            self.mask_real.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.mask_real.uniform_(-1 * mask_scale, mask_scale)
        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binariser(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternariser(threshold=threshold)

    def forward(self, input):
        # Get binarized/ternarized mask from real-valued mask.
        mask_thresholded = self.threshold_fn(self.mask_real)
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        # Get output using modified weight.
        
        print(self.weight)
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)
        
if __name__ == "__main__":
    _ = WeightMaskedLinear()
