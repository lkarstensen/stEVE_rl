from typing import Tuple
import logging
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn.functional as F
from .algo import Algo
from .model import SACModel
from ..replaybuffer import Batch


class SAC(Algo):
    def __init__(
        self,
        model: SACModel,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scaling: float = 1,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.25,
    ):
        super().__init__()
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
        # Model
        self.model = model

        # REST
        self.reward_scaling = reward_scaling
        self.action_scaling = action_scaling

        self.device = torch.device("cpu")
        self.update_step = 0

        # ENTROPY TEMPERATURE
        self.alpha = torch.ones(1)
        self.target_entropy = -torch.ones(1) * n_actions

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
            action += np.random.normal(0, self.exploration_action_noise)
        return action

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, _ = self.model.policy.forward_play(torch_state)
            action = torch.tanh(mean)
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action * self.action_scaling

    def update(self, batch: Batch) -> Tuple[float, float, float]:

        (all_states, actions, rewards, dones, padding_mask) = batch
        # actions /= self.action_scaling

        all_states = all_states.to(dtype=torch.float32, device=self.device)
        actions = actions.to(dtype=torch.float32, device=self.device)
        rewards = rewards.to(dtype=torch.float32, device=self.device)
        dones = dones.to(dtype=torch.float32, device=self.device)

        if padding_mask is not None:
            padding_mask = padding_mask.to(dtype=torch.float32, device=self.device)

        seq_length = actions.shape[1]
        states = torch.narrow(all_states, dim=1, start=0, length=seq_length)

        # use all_states for next_actions and next_log_pi for proper hidden_state initilaization
        expected_q = self._get_expected_q(
            all_states, rewards, dones, padding_mask, seq_length
        )

        # q1 update
        q1_loss = self._update_q1(actions, padding_mask, states, expected_q)

        # q2 update
        q2_loss = self._update_q2(actions, padding_mask, states, expected_q)

        log_pi, policy_loss = self._update_policy(padding_mask, states)

        self.model.update_target_q(self.tau)

        self._update_alpha(log_pi)

        self.update_step += 1
        return [
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
        ]

    def _update_alpha(self, log_pi):
        alpha_loss = (
            self.model.log_alpha * (-log_pi - self.target_entropy).detach()
        ).mean()
        self.model.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.model.alpha_optimizer.step()

        self.alpha = self.model.log_alpha.exp()

    def _update_policy(self, padding_mask, states):
        new_actions, log_pi = self._get_update_action(states)
        q1 = self.model.q1(states, new_actions)
        q2 = self.model.q2(states, new_actions)
        min_q = torch.min(q1, q2)

        if padding_mask is not None:
            min_q *= padding_mask
            log_pi *= padding_mask

        policy_loss = (self.alpha * log_pi - min_q).mean()

        self.model.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.model.policy_optimizer.step()
        return log_pi, policy_loss

    def _update_q2(self, actions, padding_mask, states, expected_q):
        curr_q2 = self.model.q2(states, actions)
        curr_q2 *= padding_mask if padding_mask is not None else curr_q2
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.model.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.model.q2_optimizer.step()
        return q2_loss

    def _update_q1(self, actions, padding_mask, states, expected_q):
        curr_q1 = self.model.q1(states, actions)
        curr_q1 *= padding_mask if padding_mask is not None else curr_q1
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())

        self.model.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.model.q1_optimizer.step()
        return q1_loss

    def _get_expected_q(self, all_states, rewards, dones, padding_mask, seq_length):
        next_actions, next_log_pi = self._get_update_action(all_states)

        with torch.no_grad():
            next_target_q1 = self.model.target_q1(all_states, next_actions)
            next_target_q2 = self.model.target_q2(all_states, next_actions)

        next_target_q = (
            torch.min(next_target_q1, next_target_q2) - self.alpha * next_log_pi
        )
        # only use next_state for next_q_target
        next_target_q = torch.narrow(next_target_q, dim=1, start=1, length=seq_length)
        expected_q = rewards + (1 - dones) * self.gamma * next_target_q
        expected_q *= padding_mask if padding_mask is not None else expected_q
        return expected_q

    # epsilon makes sure that log(0) does not occur
    def _get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_batch, log_std = self.model.policy(state_batch)
        std_batch = log_std.exp()

        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)

        # log_pi_batch = torch.sum(normal.log_prob(z), dim=-1, keepdim=True) - torch.sum(
        #        torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True)

        return action_batch, log_pi_batch

    def lr_scheduler_step(self) -> None:
        super().lr_scheduler_step()
        if self.model.q1_scheduler is not None:
            self.model.q1_scheduler.step()
        if self.model.q2_scheduler is not None:
            self.model.q2_scheduler.step()
        if self.model.policy_scheduler is not None:
            self.model.policy_scheduler.step()

    def to(self, device: torch.device):
        super().to(device)
        self.alpha = self.alpha.to(device)
        self.target_entropy = self.target_entropy.to(device)
        self.model.to(device)

    def reset(self) -> None:
        self.model.reset()

    def close(self):
        self.model.close()

    def copy_play_only(self):
        return self.__class__(
            self.model.copy_play_only(),
            self.n_actions,
            self.gamma,
            self.tau,
            self.reward_scaling,
            self.action_scaling,
            self.exploration_action_noise,
        )


class SACStochasticEval(SAC):
    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            torch_state = torch.as_tensor(
                flat_state, dtype=torch.float32, device=self.device
            )
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
            mean, log_std = self.model.policy.forward_play(torch_state)
            std = log_std.exp()
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
            action = action.squeeze(0).squeeze(0).cpu().detach().numpy()
        return action
