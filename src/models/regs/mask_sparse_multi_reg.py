from torch import nn


class MaskSparseMultiReg(nn.Module):
    """The sparsity reguralisation on masks keeps model capacity reserved for future tasks.

    See "2.6. Promoting Low Capacity Usage" in HAT paper:
    https://arxiv.org/abs/1801.01423
    """

    def __init__(self, factor: float):
        super().__init__()

        self.factor = factor  # regularisation factor

    def forward(self, mask, mask_refs):
        reg_total = {}  # reg value
        for idx, mask_ref in mask_refs.items():
            reg_total[idx] = 0
            count_total = 0
            for module_name in mask.keys():
                m = mask[module_name]
                m_ref = mask_ref[module_name]
                aux = 1 - m_ref
                reg_l = (m * aux).sum()
                reg_total[idx] += reg_l
                count_l = aux.sum()
                reg_l = reg_l / count_l
                count_total += count_l
            reg_total[idx] = reg_total[idx] / count_total
            # print(reg_total)
        return self.factor * sum(reg_total.values()), reg_total
