import random
import numpy as np
import torch
from .replaybuffer import ReplayBuffer, Batch

class SingleTuple(ReplayBuffer):
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self._batch_size = batch_size
        self.buffer = []
        self.position = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size


    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        states = np.concatenate((state, next_state), axis=-1)
        # !
        states = states.reshape((2,6))
        self.buffer[self.position] = (states, action, reward, done)
        self.position = int((self.position + 1) % self.capacity)  

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        batch = list(map(np.stack, zip(*batch)))  
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        batch = [torch.from_numpy(batch_entry) for batch_entry in batch]
        batch[1] = batch[1].unsqueeze(1)
        batch[2] = batch[2].unsqueeze(1)
        batch[3] = batch[3].unsqueeze(1)

        return Batch(*batch)


    def __len__(
        self,
    ):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(self.capacity, self.batch_size)
        for i in range(len(self.buffer)):
            copy.buffer.append(self.buffer[i])
        copy.position = self.position
        return copy

    def close(self):
        ...