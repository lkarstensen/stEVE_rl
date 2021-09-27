from math import inf
import random
from typing import List
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_sequence
from .replaybuffer import ReplayBuffer, Episode, Batch


class VanillaEpisode(ReplayBuffer):
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self._batch_size = batch_size
        self.buffer: List[Episode] = []
        self.position = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if len(episode.states) == 0:
            return
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode.to_numpy()
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self) -> Batch:
        episodes = random.sample(self.buffer, self.batch_size)

        state_batch = [torch.from_numpy(episode.states) for episode in episodes]
        action_batch = [torch.from_numpy(episode.actions) for episode in episodes]
        reward_batch = [torch.from_numpy(episode.rewards) for episode in episodes]
        next_state_batch = [torch.from_numpy(episode.next_states) for episode in episodes]
        done_batch = [torch.from_numpy(episode.dones) for episode in episodes]

        state_batch = pad_sequence(state_batch, batch_first=True)
        action_batch = pad_sequence(action_batch, batch_first=True)
        reward_batch = pad_sequence(reward_batch, batch_first=True, padding_value=inf)
        next_state_batch = pad_sequence(next_state_batch, batch_first=True)
        done_batch = pad_sequence(done_batch, batch_first=True)

        padding_mask = torch.ones_like(reward_batch)
        padding_mask[reward_batch == inf] = 0
        reward_batch[reward_batch == inf] = 0
        return Batch(
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, padding_mask
        )

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
