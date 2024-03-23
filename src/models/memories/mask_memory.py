from copy import deepcopy

import torch
from torch import nn


class MaskMemory:
    """Memory storing masks for HAT algorithm.

    Args:
        s_max (float): max scale of mask gate function
        backbone (nn.Module): only use its mask shape to create new mask
    """

    def __init__(self, s_max: float, backbone: nn.Module, approach: str):
        self.s_max = s_max
        self.backbone = backbone  # help define the data shape of masks

        self.approach = approach

        # stores masks
        self.masks = {}
        self.masks[0] = self.empty_mask()  # init for mask sparse multi reg

        # stores cumulated mask of all self.masks.
        self.union_mask = self.empty_mask()
        if self.approach == "adahat":
            self.sum_mask = self.empty_mask()

    def get_mask(self, task_id: int):
        """Get mask of task_id."""
        return self.masks[task_id]

    def get_masks(self):
        """Get all masks."""
        return self.masks

    def empty_mask(self):
        """Create empty mask (all zeros) with mask size of backbone."""
        mask = {}
        for module_name, embedding in self.backbone.te.items():
                mask[module_name] = torch.zeros_like(embedding.weight).to("cuda:0")

        return mask

    def get_union_mask(self):
        return self.union_mask

    def get_weight_mask(
        self, module_name: str, mask_type: str, view_shape, weight_size
    ):
        with torch.no_grad():
            mask = (
                self.union_mask[module_name].view(*view_shape)
                if mask_type == "union"
                else self.sum_mask[module_name].view(*view_shape)
            )

            module_order = self.backbone.module_order
            upper_module_index = module_order.index(module_name) - 1
            
            
            upper_module_name = module_order[upper_module_index] if upper_module_index != -1 else None
            # if upper_module_name == "residual": upper_module_name = None
                
            if upper_module_name: 
                upper_mask = (
                    self.union_mask[upper_module_name]
                    if mask_type == "union"
                    else self.sum_mask[upper_module_name]
                )
            else:
                upper_mask = None

            mask_expand = mask.expand(weight_size)
            # print("mask",mask.size())
            if upper_module_name:
                # print("upper_mask", upper_mask.size())
                if upper_mask.dim() != len(weight_size):
                    for i in range(upper_mask.dim(), len(weight_size)):
                        # print("i")
                        upper_mask = upper_mask.unsqueeze(-1)
                # print(upper_mask.size())
                upper_mask_expand = upper_mask.expand(weight_size)
                weight_mask = torch.min(mask_expand, upper_mask_expand)
            else:
                weight_mask = mask_expand
                # print("weight_mask", weight_mask.size())            

        return weight_mask

    def get_sum_mask(self):
        return self.sum_mask

    def combine_masks(self, mask1, mask2, operator="unite"):
        """Join two masks by element-wise maximum."""
        mask = {}
        # for m in self.backbone.modules():
        #     print(m)

        for module_name in mask1.keys():
            if operator == "union":
                mask[module_name] = torch.max(mask1[module_name], mask2[module_name])
            elif operator == "sum":
                mask[module_name] = mask1[module_name] + mask2[module_name]
        return mask

    def update_union_mask(self, mask):
        """Update cumulated union mask."""
        union_mask = deepcopy(self.union_mask)
        self.union_mask = self.combine_masks(union_mask, mask, operator="union")

    def update_sum_mask(self, mask):
        """Update cumulated sum mask."""
        sum_mask = deepcopy(self.sum_mask)
        self.sum_mask = self.combine_masks(sum_mask, mask, operator="sum")

    def te2mask(self, te, backbone):
        mask = {}
        for module_name, embedding in te.items():
            mask[module_name] = backbone.mask(embedding, self.s_max).detach()
        return mask

    def update(self, task_id: int, backbone: torch.nn.Module):
        """Store mask of self.task_id after training it, and update union mask."""
        mask = self.te2mask(backbone.te, backbone)
        self.masks[task_id] = mask
        self.update_union_mask(mask)
        if self.approach == "adahat":
            self.update_sum_mask(mask)


if __name__ == "__main__":
    # import pyrootutils
    # pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    # from src.models.backbones import MaskedMLP
    # backbone = MaskedMLP(input_dim=784, hidden_dims=[256,256], output_dim=64)
    # mask_memory = MaskMemory(s_max=10, backbone=backbone)
    # print(backbone.te["fc1"].weight)
    # print(mask_memory.get_union_mask())
    # mask_memory.update(task_id=0, backbone=backbone)
    # print(mask_memory.get_union_mask())
    # print(mask_memory.get_mask(0))
    mask = {"fc1": torch.tensor([])}
