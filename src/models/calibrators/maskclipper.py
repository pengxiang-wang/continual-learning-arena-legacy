import random

import torch
from torch import nn


def hard_clip_te_masked_gradients(backbone: nn.Module, mask_memory, log_capacity: bool):
    """fix te from previous mask by hard clip their gradients to zero."""
    
    if log_capacity:    
        capacity = 0
        num = 0

    union_mask = mask_memory.get_union_mask()
    for module_name, module in backbone.named_modules():

        module_name = module_name.replace(".", "")

        if module_name in union_mask.keys():
            # with torch.no_grad():
            

            view_shape = [1 for i in range(module.weight.grad.data.dim())]
            view_shape[0] = -1
            weight_size = module.weight.size()
            weight_mask = mask_memory.get_weight_mask(
                module_name, "union", view_shape, weight_size
            )


            adjust = 1 - weight_mask
            adjust_bias = 1 - torch.squeeze(union_mask[module_name])

            module.weight.grad.data *= adjust #1 - union_mask[module_name].view(*view_shape)
            module.bias.grad.data *= adjust_bias

            if log_capacity:
                capacity += torch.sum(adjust)
                num += torch.sum(torch.ones_like(adjust))
                capacity += torch.sum(adjust_bias)
                num += torch.sum(torch.ones_like(adjust_bias))

        capacity = capacity / num if log_capacity else None

    return capacity


def soft_clip_te_masked_gradients(
    backbone: nn.Module,
    adjust_strategy: str,
    mask_memory,
    task_id: int,
    mask_sparse_loss,
    alpha,
    log_capacity: bool,
):
    """fix te from previous mask by soft clip their gradients according to masked links."""
    if log_capacity:    
        capacity = 0
        num = 0

    # with torch.no_grad():

    union_mask = mask_memory.get_union_mask()
    sum_mask = mask_memory.get_sum_mask()

    for module_name, module in backbone.named_modules():

        module_name = module_name.replace(".", "")

        if module_name in union_mask.keys():
            view_shape = [1 for i in range(module.weight.grad.data.dim())]
            view_shape[0] = -1

            weight_size = module.weight.size()

            weight_mask = mask_memory.get_weight_mask(
                module_name, "union", view_shape, weight_size
            )
            weight_sum_mask = mask_memory.get_weight_mask(
                module_name, "sum", view_shape, weight_size
            )

            if adjust_strategy == "ada_prob":
                if random.random() < (1 - mask_sparse_loss[module_name]):
                    # print(mask_sparse_loss)
                    r = alpha  # / (0.1 + mask_sparse_loss[module_name])
                    # print(adjust.size())
                    adjust = torch.div(r, (weight_sum_mask + r))
                    adjust_bias = torch.div(
                        r, (torch.squeeze(sum_mask[module_name]) + r)
                    )
                else:
                    adjust = 1 - weight_mask
                    adjust_bias = 1 - torch.squeeze(union_mask[module_name])

            elif adjust_strategy == "ada":
                r = alpha / (0.1 + mask_sparse_loss[module_name])
                adjust = torch.div(r, (weight_sum_mask + r))
                adjust_bias = torch.div(
                    r, (torch.squeeze(sum_mask[module_name]) + r)
                )

            elif adjust_strategy == "ada_ave":
                r = alpha / (0.1 + mask_sparse_loss[module_name])
                if task_id == 0:
                    adjust = torch.tensor(1)
                    adjust_bias = torch.tensor(1)
                else:
                    adjust = torch.div(r, (weight_sum_mask / task_id + r))
                    adjust_bias = torch.div(
                        r, (torch.squeeze(sum_mask[module_name]) / task_id + r)
                    )

            elif adjust_strategy == "ada_ave_prob":
                if random.random() < (1 - mask_sparse_loss[module_name]):
                    # print(mask_sparse_loss)
                    r = alpha  # / (0.1 + mask_sparse_loss[module_name])
                    # print(adjust.size())
                    if task_id == 0:
                        adjust = torch.tensor(1)
                        adjust_bias = torch.tensor(1)
                    else:
                        adjust = torch.div(r, (weight_sum_mask / task_id + r))
                        adjust_bias = torch.div(
                            r, (torch.squeeze(sum_mask[module_name]) / task_id + r)
                        )

                else:
                    adjust = 1 - weight_mask
                    adjust_bias = 1 - torch.squeeze(union_mask[module_name])

            elif adjust_strategy == "ada_sum_1":
                r = alpha / (0.1 + mask_sparse_loss[module_name])
                adjust = torch.div(r, (1 + r))
                adjust_bias = torch.div(r, (1 + r))

            elif adjust_strategy == "ada_sum_t":
                r = alpha / (0.1 + mask_sparse_loss[module_name])
                adjust = torch.div(r, (task_id + r))
                adjust_bias = torch.div(r, (task_id + r))

            elif adjust_strategy == "ada_reg_1":
                r = alpha / (0.1 + 1)
                adjust = torch.div(r, (weight_sum_mask + r))
                adjust_bias = torch.div(
                    r, (torch.squeeze(sum_mask[module_name]) + r)
                )

            elif adjust_strategy == "ada_reg_09":
                r = alpha
                adjust = torch.div(r, (weight_sum_mask + r))
                adjust_bias = torch.div(
                    r, (torch.squeeze(sum_mask[module_name]) + r)
                )

            elif adjust_strategy == "ada_reg_0":
                r = alpha / (0.1 + 0)
                adjust = torch.div(r, (weight_sum_mask + r))
                adjust_bias = torch.div(
                    r, (torch.squeeze(sum_mask[module_name]) + r)
                )

            elif adjust_strategy == "ada_random":
                adjust = random.random() * (1 - weight_mask)
                adjust_bias = random.random() * (
                    1 - torch.squeeze(union_mask[module_name])
                )

            elif adjust_strategy == "ada_random_all":
                adjust = torch.tensor(random.random())
                adjust_bias = torch.tensor(random.random())

            elif adjust_strategy == "ada_cons_alpha_all":
                adjust = torch.tensor(alpha)
                adjust_bias = torch.tensor(alpha)

            elif adjust_strategy == "ada_cons_alpha":
                adjust = alpha * (1 - weight_mask)
                adjust_bias = alpha * (1 - torch.squeeze(union_mask[module_name]))

            elif adjust_strategy == "ada_cons_1":
                adjust = torch.tensor(1)
                adjust_bias = torch.tensor(1)

            else:
                adjust = None
                adjust_bias = None

            module.weight.grad.data *= adjust
            module.bias.grad.data *= adjust_bias

            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # adjust = adjust.to(device)
            # adjust_bias = adjust_bias.to(device)

            # print(adjust, float(torch.sum(adjust)))
            if log_capacity:

                capacity += float(torch.sum(adjust))
                num += float(torch.sum(torch.ones_like(adjust)))
                capacity += float(torch.sum(adjust_bias))
                num += float(torch.sum(torch.ones_like(adjust_bias)))

        capacity = capacity / num if log_capacity else None

    return capacity


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
