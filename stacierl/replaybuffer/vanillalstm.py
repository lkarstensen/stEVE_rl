import random
from typing import List
import numpy as np
from .replaybuffer import ReplayBuffer, Episode, Batch


class VanillaLSTM(ReplayBuffer):
    def __init__(self, capacity, sequence_length: int):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer: List[Episode] = []
        self.position = 0

    def push(self, episode: Episode):

        for i in range(len(episode)):
            episode.hidden_states[i] = (
                episode.hidden_states[i][0].cpu().numpy().squeeze(1),
                episode.hidden_states[i][1].cpu().numpy().squeeze(1),
            )

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size: int) -> Batch:
        episodes = random.sample(self.buffer, batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        hidden_state_batch = []
        hidden_next_state_batch = []
        cell_state_batch = []
        cell_next_state_batch = []
        for episode in episodes:
            state_seq = episode.states
            action_seq = episode.actions
            reward_seq = episode.rewards
            next_state_seq = episode.next_states
            done_seq = episode.dones
            hidden_state_seq = episode.hidden_states

            if len(episode) == 1:
                continue

            elif len(episode) >= self.sequence_length + 1:
                start_idx = np.random.randint(0, len(episode) - self.sequence_length)
                end_idx = start_idx + self.sequence_length

                state_seq = state_seq[start_idx:end_idx]
                action_seq = action_seq[start_idx:end_idx]
                reward_seq = reward_seq[start_idx:end_idx]
                next_state_seq = next_state_seq[start_idx:end_idx]
                done_seq = done_seq[start_idx:end_idx]
                hidden_state = hidden_state_seq[start_idx]
                hidden_next_state = hidden_state_seq[start_idx + 1]

            else:
                n_padding = self.sequence_length - len(episode)

                state_padding_shape = tuple([n_padding]) + state_seq[0].shape
                state_padding = np.full(state_padding_shape, 0.0)
                state_seq = np.concatenate((state_padding, state_seq), axis=0)
                next_state_seq = np.concatenate((state_padding, next_state_seq), axis=0)

                action_padding_shape = tuple([n_padding]) + action_seq[0].shape
                action_padding = np.full(action_padding_shape, 0.0)
                action_seq = np.concatenate((action_padding, action_seq), axis=0)

                reward_seq = [0] * n_padding + reward_seq

                done_seq = [False] * n_padding + done_seq
                hidden_state = hidden_state_seq[0]
                hidden_next_state = hidden_state_seq[1]
            state_batch.append(state_seq)
            action_batch.append(action_seq)
            reward_batch.append(reward_seq)
            next_state_batch.append(next_state_seq)
            done_batch.append(done_seq)
            hidden_state_batch.append(hidden_state[0])
            hidden_next_state_batch.append(hidden_next_state[0])
            cell_state_batch.append(hidden_state[1])
            cell_next_state_batch.append(hidden_next_state[1])
        state_batch = np.asarray(state_batch)
        action_batch = np.asarray(action_batch)
        reward_batch = np.asarray(reward_batch)
        next_state_batch = np.asarray(next_state_batch)
        done_batch = np.asarray(done_batch)
        hidden_state_batch = np.asarray(hidden_state_batch)
        hidden_state_batch = np.swapaxes(hidden_state_batch, 0, 1).copy()
        hidden_next_state_batch = np.asarray(hidden_next_state_batch)
        hidden_next_state_batch = np.swapaxes(hidden_next_state_batch, 0, 1).copy()
        cell_state_batch = np.asarray(cell_state_batch)
        cell_state_batch = np.swapaxes(cell_state_batch, 0, 1).copy()
        cell_next_state_batch = np.asarray(cell_next_state_batch)
        cell_next_state_batch = np.swapaxes(cell_next_state_batch, 0, 1).copy()
        return Batch(
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            hidden_state_batch,
            hidden_next_state_batch,
            cell_state_batch,
            cell_next_state_batch,
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
