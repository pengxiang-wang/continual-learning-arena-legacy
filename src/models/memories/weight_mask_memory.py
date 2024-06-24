from copy import deepcopy

import torch
from torch import nn
from models.backbones.functions import Binariser


class WeightMaskMemory:
    """Memory storing whole models (networks) for previous tasks.

    Args:
        s_max (float): max scale of mask gate function
    """

    def __init__(self, backbone: nn.Module, datatype, threshold):
        self.backbone = backbone  # help define the data shape of masks
        self.datatype = datatype
        # stores masks
        self.masks = {}

        self.binariser = Binariser(threshold=threshold)

        # stores cumulated mask of all self.masks.
        self.union_mask = self.empty_mask()

    def get_mask(self, task_id: int):
        """Get mask of task_id."""
        return self.masks[task_id]

    def empty_mask(self):

        mask = {}

        for n, p in self.backbone.named_parameters():
            mask[n] = 0 * p.data

        return mask

    def real2binary(self, real_mask):
        binary_mask = {}
        for n, p in self.backbone.named_parameters():
            binary_mask[n] = self.binariser(real_mask[n])
        return binary_mask

    def update(self, task_id: int, backbone: torch.nn.Module):
        """Store model (including backbone and heads) of self.task_id after training it."""
        self.masks[task_id] = backbone.mask

        print(backbone.mask)

        binary_mask = self.real2binary(backbone.mask)

        print(binary_mask)

        self.masks[task_id] = binary_mask
        # for module_idx, module in enumerate(backbone.modules()):
        #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

        #         if datatype == "binary":
        #             m = torch.ByteTensor(module.weight.data.size()).fill_(1)
        #         elif datatype == "real":
        #             m = torch.Tensor(module.weight.data.size()).fill_(1)

        #         if 'cuda' in module.weight.data.type():
        #             mask = mask.cuda()
        #         mask[module_idx] = m

        # mask = {}
        # for param_name, value in backbone.state_dict().items():
        #     mask[param_name] = torch.zeros_like(value)

        # return mask

    # def get_union_mask(self):
    #     return self.union_mask

    # def combine_masks(self, mask1, mask2):
    #     """Join two masks by element-wise maximum."""
    #     mask = {}
    #     for param_name in mask1.keys():
    #         mask[param_name] = torch.max(mask1[param_name], mask2[param_name])
    #     return mask

    # def update_union_mask(self, mask):
    #     """Update cumulated union mask."""
    #     union_mask = deepcopy(self.union_mask)
    #     self.union_mask = self.combine_masks(union_mask, mask, operator="union")


if __name__ == "__main__":
    pass
