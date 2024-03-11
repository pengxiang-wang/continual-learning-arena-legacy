import random

import torch
from torch import nn


def hard_clip_te_masked_gradients(backbone: nn.Module, union_mask):
    """fix te from previous mask by hard clip their gradients to zero."""
    for module_name, module in backbone.named_modules():
        module_name = module_name.replace(".", "")
        if module_name in union_mask.keys():
            view_shape = [1 for i in range(module.weight.grad.data.dim())]
            view_shape[0] = -1
            module.weight.grad.data *= 1 - union_mask[module_name].view(*view_shape)


def soft_clip_te_masked_gradients(
    adjust_strategy: str, backbone: nn.Module, sum_mask, union_mask, mask_sparse_loss, alpha
):
    """fix te from previous mask by soft clip their gradients according to masked links."""
    for module_name, module in backbone.named_modules():
        module_name = module_name.replace(".", "")
        
        if module_name in sum_mask.keys():
            # print(mask_sparse_loss[module_name])
            view_shape = [1 for i in range(module.weight.grad.data.dim())]
            view_shape[0] = -1

            if adjust_strategy == "original":
                if random.random() < (1 - mask_sparse_loss[module_name]):
                    # print(mask_sparse_loss)
                    r = alpha # / (0.1 + mask_sparse_loss[module_name])
                    # print(adjust.size())
                    adjust = torch.div(
                        r, (sum_mask[module_name].view(*view_shape) + r)
                    )
                else:
                    adjust = 1 - union_mask[module_name].view(*view_shape)
                    
            elif adjust_strategy == "ada":
                    r = alpha / (0.1 + mask_sparse_loss[module_name])
                    adjust = torch.div(
                        r, (sum_mask[module_name].view(*view_shape) + r)
                    )

            elif adjust_strategy == "ada_no_sum":
                    r = alpha / (0.1 + mask_sparse_loss[module_name])
                    adjust = torch.div(
                        r, (1 + r)
                    )
                    
            elif adjust_strategy == "ada_no_reg":
                    r = alpha / (0.1 + 1)
                    adjust = torch.div(
                        r, (sum_mask[module_name].view(*view_shape) + r)
                    )
                    
            elif adjust_strategy == "random":
                adjust = random.random()
                
            elif adjust_strategy == "constant":
                adjust = alpha
                
            else:
                adjust = None
                    
            module.weight.grad.data *= adjust
            
        # return adjust # for visualisation


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
