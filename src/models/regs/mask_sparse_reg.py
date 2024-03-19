from torch import nn


class MaskSparseReg(nn.Module):
    """The sparsity reguralisation on masks keeps model capacity reserved for future tasks.

    See "2.6. Promoting Low Capacity Usage" in HAT paper:
    https://arxiv.org/abs/1801.01423
    """

    def __init__(self, factor: float, type="old"):
        super().__init__()

        self.type = type

        self.factor = factor  # regularisation factor

    def forward(self, mask, mask_ref):
        reg1_total = 0  # reg value
        reg2_total = 0
        reg = {}  # reg value from each part of mask
        count1_total = 0
        count2_total = 0
        for module_name in mask.keys():
            m = mask[module_name]
            m_ref = mask_ref[module_name]
            m_ref_reverse = 1 - m_ref

            reg1_l = (m * m_ref_reverse).sum()
            reg1_total += reg1_l
            count1_l = m_ref_reverse.sum()
            count1_total += count1_l
            reg1_l = reg1_l / count1_l if count1_l > 10 else 0

            reg2_l = (m * m_ref).sum()
            reg2_total += reg2_l
            count2_l = m_ref.sum()
            count2_total += count2_l
            reg2_l = 1 - reg2_l / count2_l if count2_l > 10 else 1

            reg_l = 0.5 * (reg1_l + reg2_l)
            reg[module_name] = reg_l

            # print(f"reg1{module_name}", reg1_l)
            # print(f"reg2{module_name}", reg2_l)
            # print(f"reg{module_name}", reg_l)

        reg1_total = reg1_total / count1_total if count1_total > 10 else 0
        reg2_total = 1 - reg2_total / count2_total if count2_total > 10 else 1

        if self.type == "new":
            reg_total = 0.5 * (reg1_total + reg2_total)
        elif self.type == "old":
            reg_total = reg1_total
        # print(reg_total)
        return self.factor * reg_total, reg_total, reg
