from math import inf
import random
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from .replaybuffer import ReplayBuffer, Episode, Batch
import numpy as np


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
        if len(episode) < 1:
            return
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        episode_np = (
            np.array(episode.flat_states),
            np.array(episode.actions),
            np.array(episode.rewards),
            np.array(episode.terminals),
        )
        self.buffer[self.position] = episode_np
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self) -> Batch:
        episodes = random.sample(self.buffer, self.batch_size)

        state_batch = [torch.from_numpy(episode[0]) for episode in episodes]
        action_batch = [torch.from_numpy(episode[1]) for episode in episodes]
        reward_batch = [
            torch.from_numpy(episode[2]).unsqueeze(1) for episode in episodes
        ]
        done_batch = [torch.from_numpy(episode[3]).unsqueeze(1) for episode in episodes]

        state_batch = pad_sequence(state_batch, batch_first=True)
        action_batch = pad_sequence(action_batch, batch_first=True)
        reward_batch = pad_sequence(reward_batch, batch_first=True, padding_value=inf)
        done_batch = pad_sequence(done_batch, batch_first=True)

        padding_mask = torch.ones_like(reward_batch)
        padding_mask[reward_batch == inf] = 0
        reward_batch[reward_batch == inf] = 0
        return Batch(state_batch, action_batch, reward_batch, done_batch, padding_mask)

    def __len__(self):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(self.capacity, self.batch_size)
        for i in range(len(self.buffer)):
            copy.buffer.append(self.buffer[i])
        copy.position = self.position
        return copy

    def close(self):
        del self.buffer
