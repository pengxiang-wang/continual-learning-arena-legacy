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
        
        # return nn.CrossEntropyLoss()(y_student, y_teacher)
        
        y_student = nn.functional.softmax(y_student, dim=1)
        y_teacher = nn.functional.softmax(y_teacher, dim=1)
        
        # print("y_student", y_student.size())
        # print("y_teacher", y_teacher.size())
        
        y_student = y_student.pow(1/self.temp)
        y_teacher = y_teacher.pow(1/self.temp)

        y_student = torch.div(y_student, torch.sum(y_student, 1, keepdim=True))
        y_teacher = torch.div(y_teacher ,torch.sum( y_teacher,1, keepdim=True ))
        
        y_student = y_student + 1e-5 / y_student.size(1)
        y_student = torch.div(y_student, torch.sum(y_student,1, keepdim=True ))
        

        
        ce = - (y_teacher * y_student.log()).sum(1) # don't use nn.CrossEntropy, it contains Softmax

        ce = ce.mean()
        return self.factor * ce
