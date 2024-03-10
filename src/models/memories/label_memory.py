import torch

import random


class LabelMemory:
    """Memory storing data labels from previous tasks.
    
    It might be used for the random method.
    """

    def __init__(self):
        
        # stores labels
        self.labels = {}
        
    def get_labels(self, task_id: int):
        return self.labels[task_id]
        
        
    def get_random_labels(self, task_id: int, num: int):
        
        whole_labels = self.get_labels(task_id)
        labels = torch.stack(random.sample(whole_labels, num), 0)
        return labels
        
        
        
    def update(self, batch: torch.Tensor, task_id: int):
        """Store data from self.task_id."""
        if task_id not in self.labels.keys():
            self.labels[task_id] = []
            
        _, y = batch
        
        self.labels[task_id] += y


if __name__ == "__main__":
    pass
