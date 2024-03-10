from torchvision.datasets import MNIST as OrigDataset    
    
    
class OrigDatasetTaskLabeled(OrigDataset):

    def __init__(self, task_id: int, *args, **kw):
        
        super().__init__(*args, **kw)
        self.task_label = task_id
        self.__class__.__name__ = Dataset.__name__

    def __getitem__(self, index: int):
        
        X, y = super().__getitem__(index)
        return X, y, self.task_label
    
    
    
A = OrigDatasetTaskLabeled(root='/home/bq24744/continual-learning-arena/data',
            train=False,
            download=True,task_id=1)
print(A[1])
x, y, t = A[1]
print(x, y, t)