from copy import deepcopy
import torch


class MaskMemory:
    """Memory storing masks for HAT algorithm.

    Args:
        s_max (float): max scale of mask gate function
        backbone (nn.Module): only use its mask shape to create new mask
    """

    def __init__(self, s_max: float, backbone):
        self.s_max = s_max
        self.backbone = backbone

        # stores
        self.masks = {}

        # stores joint mask of all self.masks cumulatively.
        self.cumulative_mask = self.empty_mask(backbone)

    def get_mask(self, task_id: int):
        """Get mask of task_id."""
        return self.masks[task_id]

    def empty_mask(self, backbone):
        """Create empty mask (all zeros) with mask size of backbone."""
        mask = {}
        for module_name, embedding in backbone.te.items():
            mask[module_name] = torch.zeros_like(embedding.weight)

        return mask

    def get_cumulative_mask(self):
        return self.cumulative_mask

    def join_masks(self, mask1, mask2):
        """Join two masks by element-wise maximum."""
        joint_mask = {}
        for module_name in mask1.keys():
            joint_mask[module_name] = torch.max(mask1[module_name], mask2[module_name])

        return joint_mask

    def cumulate_mask(self, mask):
        """Update cumulative mask."""
        # print(self.cumulative_mask)
        # print(mask)
        cumulative_mask = deepcopy(self.cumulative_mask)
        self.cumulative_mask = self.join_masks(cumulative_mask, mask)

        # print(self.cumulative_mask)

    def te2mask(self, te, backbone):
        mask = {}
        for module_name, embedding in te.items():
            mask[module_name] = backbone.mask(embedding, self.s_max).detach()

        # print(mask["fc1"])
        return mask

    def update(self, task_id: int, backbone: torch.nn.Module):
        """Store mask of self.task_id after training it, and update cumulative mask."""
        # print(backbone.te['fc1'].weight)
        mask = self.te2mask(backbone.te, backbone)
        self.masks[task_id] = mask

        self.cumulate_mask(mask)

        # print(self.masks)


if __name__ == "__main__":
    # import pyrootutils
    # pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    # from src.models.backbones import MaskedMLP
    # backbone = MaskedMLP(input_size=784, hidden_size=[256,256], output_size=64)
    # mask_memory = MaskMemory(s_max=10, backbone=backbone)
    # print(backbone.te["fc1"].weight)
    # print(mask_memory.get_cumulative_mask())
    # mask_memory.update(task_id=0, backbone=backbone)
    # print(mask_memory.get_cumulative_mask())
    # print(mask_memory.get_mask(0))
    mask = {"fc1": torch.tensor([])}
