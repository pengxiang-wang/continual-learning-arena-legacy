import random

import torch
from torch import nn

def hard_clip_weight_masked_gradients(backbone: nn.Module, union_mask):
    """fix te from previous mask by hard clip their gradients to zero."""

    for module_idx, module in enumerate(backbone.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            layer_mask = union_mask[module_idx]
            if module.weight.grad is not None:
                module.weight.grad.data[layer_mask.ne(
                    1)] = 0


# def hard_clip_weight_masked_gradients(backbone: nn.Module, union_mask):
#     for param_name, value in union_mask.items():
#         print(backbone.state_dict()[param_name])
#         backbone.state_dict()[param_name].grad.data *= 1 - value
         
def pruning_mask(weights, union_mask, layer_idx, prune_perc):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        union_mask = union_mask
        tensor = weights[union_mask.eq(1)]
        abs_tensor = tensor.abs()
        cutoff_rank = round(prune_perc * tensor.numel())
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0][0]

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * \
            union_mask.eq(1)

        # mask = 1 - remove_mask
        union_mask[remove_mask.eq(1)] = 0
        mask = union_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        return mask    

         
def prune(backbone: nn.Module, union_mask, task_id, prune_perc):
    """Gets pruning mask for each layer, based on previous_masks.
        Sets the self.current_masks to the computed pruning masks.
    """
    print('Pruning for dataset idx: %d' % (task_id))
    current_mask = {}

    print('Pruning each layer by removing %.2f%% of values' %
            (100 * prune_perc))
    for module_idx, module in enumerate(backbone.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = pruning_mask(
                module.weight.data, union_mask[module_idx], module_idx, prune_perc)
            current_mask[module_idx] = mask
            # Set pruned weights to 0.
            weight = module.weight.data
            weight[current_mask[module_idx].eq(0)] = 0.0
    
    return current_mask
    
                
# def prune_masked_weight(backbone: nn.Module, union_mask, prune_perc):
#     for param_name, value in union_mask.items():
#         count = sum(value)
#         count_after_pruned = count * prune_perc
    
#     return None
    #     sorter = deepcopy(value)
    #     for i in value:
    #         if value[i] == 1:
    #             sorter[i] = - inf
                
                
            
            
    #     backbone.state_dict()[param_name]
    # _, idx = torch.sort(sorter)
    # # idx to mask
    # mask = 

    # # set to zero
    # if not in idx:
    #     backbone.state_dict()[param_name].item = 0

    
    return mask