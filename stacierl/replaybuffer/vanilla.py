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
            if episode.hidden_states:
                if isinstance(episode.hidden_states[i], tuple):
                    episode.hidden_states[i] = (
                        episode.hidden_states[i][0].cpu().numpy().squeeze(1),
                        episode.hidden_states[i][1].cpu().numpy().squeeze(1),
                    )
                    episode.next_hidden_states[i] = (
                        episode.next_hidden_states[i][0].cpu().numpy().squeeze(1),
                        episode.next_hidden_states[i][1].cpu().numpy().squeeze(1),
                    )
                else:
                    episode.hidden_states[i] = (episode.hidden_states[i].cpu().numpy().squeeze(1),)
                    episode.next_hidden_states[i] = (
                        episode.next_hidden_states[i].cpu().numpy().squeeze(1),
                    )
                self.buffer[self.position] = self.buffer[self.position] + tuple(
                    [episode.hidden_states[i], episode.next_hidden_states[i]]
                )
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.buffer, batch_size)

        batch = list(map(np.stack, zip(*batch)))  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        if len(batch) == 5:
            return Batch(*batch)
        else:
            state_batch = np.expand_dims(batch[0], axis=1)
            action_batch = np.expand_dims(batch[1], axis=1)
            reward_batch = np.expand_dims(batch[2], axis=1)
            next_state_batch = np.expand_dims(batch[3], axis=1)
            done_batch = np.expand_dims(batch[4], axis=1)
            hidden_state_batch = np.asarray(batch[5])
            hidden_state_batch = np.moveaxis(hidden_state_batch, 0, -2).copy()
            hidden_next_state_batch = np.asarray(batch[6])
            hidden_next_state_batch = np.moveaxis(hidden_next_state_batch, 0, -2).copy()

            if len(hidden_state_batch.shape) > 3:
                hidden_state_batch = (hidden_state_batch[0], hidden_state_batch[1])
                next_hidden_state_batch = (hidden_next_state_batch[0], hidden_next_state_batch[1])

            return Batch(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch,
                hidden_state_batch,
                next_hidden_state_batch,
            )

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

    def close(self):
        ...
