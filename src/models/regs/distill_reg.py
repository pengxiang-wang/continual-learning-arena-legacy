from torch import nn


class DistillReg(nn.Module):
    """The distillation regularisation.

    See in Knowledge Distillation paper:
    https://arxiv.org/abs/1503.02531
    """

    def __init__(self, factor: float, temp: float):
        super().__init__()

        self.factor = factor  # regularisation factor

        self.temp = temp
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()


    def forward(self, y_student, y_teacher):
        
        y_student_soften = self.softmax(y_student / self.temp, dim=1)
        
        y_teacher_soften = self.softmax(y_teacher / self.temp, dim=1)
        
        return self.loss_fn(y_student_soften, y_teacher_soften)
        
