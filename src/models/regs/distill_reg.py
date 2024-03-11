from torch import nn
import torch

class DistillReg(nn.Module):
    """The distillation regularisation.

    See in Knowledge Distillation paper:
    https://arxiv.org/abs/1503.02531
    """

    def __init__(self, factor: float, temp: float):
        super().__init__()

        self.factor = factor  # regularisation factor

        self.temp = temp


    def forward(self, y_student, y_teacher):
        
        y_student = nn.functional.softmax(y_student)
        y_teacher = nn.functional.softmax(y_teacher)
        

        y_student = y_student.pow(1/self.temp)
        y_teacher = y_teacher.pow(1/self.temp)

        y_student = y_student / torch.sum(y_student)
        y_teacher = y_teacher / torch.sum(y_teacher)
        
        y_teacher = y_teacher + 1e-5 / y_teacher.size(1)
        y_teacher = y_teacher / torch.sum(y_teacher)
        
        ce = - torch.sum(y_student * y_teacher.log()) # don't use nn.CrossEntropy, it contains Softmax


        return self.factor * ce
