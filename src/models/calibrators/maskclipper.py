import random

import torch
from torch import nn


def hard_clip_te_masked_gradients(backbone: nn.Module, union_mask):
    """fix te from previous mask by hard clip their gradients to zero."""
    for module_name, module in backbone.named_modules():
        module_name = module_name.replace(".", "")
        if module_name in union_mask.keys():
            module.weight.grad.data *= 1 - union_mask[module_name].transpose(0, 1)


def soft_clip_te_masked_gradients(
    backbone: nn.Module, sum_mask, union_mask, mask_sparse_loss, adjust_rate
):
    """fix te from previous mask by soft clip their gradients according to masked links."""
    for module_name, module in backbone.named_modules():
        module_name = module_name.replace(".", "")

        if module_name in sum_mask.keys():
            # print(mask_sparse_loss[module_name])
            if random.random() < 0.5 * (1 - mask_sparse_loss[module_name]):
                # print(mask_sparse_loss)
                adjust = adjust_rate / (0.1 + mask_sparse_loss[module_name])
                # print(adjust.size())
                factor = torch.div(
                    adjust, (sum_mask[module_name].transpose(0, 1) + adjust)
                )
            else:
                factor = 1 - union_mask[module_name].transpose(0, 1)
            # print(factor[0])
            module.weight.grad.data *= factor


def compensate_te_gradients(backbone: nn.Module, compensate_thres, scalar, s_max):
    """Compensate (clamp) task embedding gradients."""
    for embedding in backbone.te.values():
        num = (
            torch.cosh(
                torch.clamp(
                    scalar * embedding.weight.data,
                    -compensate_thres,
                    compensate_thres,
                )
            )
            + 1
        )
        den = torch.cosh(embedding.weight.data) + 1
        embedding.weight.grad.data *= s_max / scalar * num / den
