from torch import nn
from copy import deepcopy


class TrapsNMineReg(nn.Module):
    """The traps and mines reguralisation on masks.
    """

    def __init__(self, mask_counter, N, factor):
        super().__init__()
        self.laziness = deepcopy(mask_counter)
        for module_name in mask_counter.keys():
            self.laziness[module_name] = self.laziness[module_name] / N
            
        self.factor = factor

    def forward(self, mask):
        for module_name in mask.keys():
            m = mask[module_name]
            cond = detach(m)
            reg += cond * m + (1 - cond) * (1 - m)
    
        return reg