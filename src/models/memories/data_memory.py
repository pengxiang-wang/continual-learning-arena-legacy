import torch


class AllDataMemory:
    """Memory storing data from previous tasks.

    It might be used for:
    1. Replay data
    2. Whole data (only for use after training)
    """

    def __init__(self):

        # stores data
        self.data = {}

    def get_data(self, task_id: int):
        return self.data[task_id]

    def release(self, task_id):
        for t in task_id:
            self.data[t] = []

    def update(self, batch: torch.Tensor, task_id: int):
        """Store data from self.task_id."""
        if task_id not in self.data.keys():
            self.data[task_id] = []
        self.data[task_id].append(batch)



class DataMemory:
    """Memory storing data from previous tasks.

    It might be used for:
    1. Replay data
    2. Whole data (only for use after training)
    """

    def __init__(self):

        # stores data
        self.data = []

    def get_data(self):
        return self.data
    
    def release(self):
        self.data = []


    def update(self, batch: torch.Tensor, task_id: int):
        """Store data from self.task_id."""
        self.data.append(batch)



if __name__ == "__main__":
    pass
