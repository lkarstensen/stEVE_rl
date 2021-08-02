import random
import numpy as np
from .replaybuffer import ReplayBuffer, Episode, Batch


class Vanilla(ReplayBuffer):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, episode: Episode):

        for i in range(len(episode)):

            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (
                episode.states[i],
                episode.actions[i],
                episode.rewards[i],
                episode.next_states[i],
                episode.dones[i],
            )
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(
            np.stack, zip(*batch)
        )  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return Batch(state, action, reward, next_state, done)

    def __len__(
        self,
    ):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(self.capacity)
        for i in range(len(self.buffer)):
            copy.buffer.append(self.buffer[i])
        copy.position = self.position
        return copy
