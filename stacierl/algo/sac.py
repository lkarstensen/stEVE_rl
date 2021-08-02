from typing import Tuple
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from .algo import Algo
from .. import model
import numpy as np
from ..replaybuffer import Batch


class SAC(Algo):
    def __init__(
        self,
        model: model.SAC,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scaling: float = 1,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device

        # HYPERPARAMETERS
        self.gamma = gamma
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
        # Model
        self.model = model

        # REST
        self.reward_scaling = reward_scaling
        self.action_scaling = action_scaling
        self.update_step = 0

        # ENTROPY TEMPERATURE
        self.alpha = 0.0
        self.target_entropy = -self.model.policy_net.n_actions

        self.model.to(device)

    def get_exploration_action(self, flat_state: np.ndarray):

        action = self.get_eval_action(flat_state)
        action += np.random.normal(0, self.exploration_action_noise)
        return action

    def get_eval_action(self, flat_state: np.ndarray):
        flat_state = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)

        mean, log_std = self.model.policy_net.forward(flat_state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        action *= self.action_scaling
        return action

        # epsilon makes sure that log(0) does not occur

    def _evaluate_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_batch, log_std = self.model.policy_net.forward(state_batch)
        std_batch = log_std.exp()

        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(1, keepdim=True)

        return action_batch, log_pi_batch, mean_batch, std_batch

    def update(self, batch: Batch):

        states, actions, rewards, next_states, dones, _ = batch
        # actions /= self.action_scaling

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = torch.reshape(rewards, (-1, 1))
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = torch.reshape(dones, (-1, 1))

        next_actions, next_log_pi, _, _ = self._evaluate_action(next_states)
        next_q1 = self.model.target_q_net_1.forward(next_states, next_actions)
        next_q2 = self.model.target_q_net_2.forward(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = (
            rewards + (1 - dones) * self.gamma * next_q_target
        )  # self.reward_scaling * rewards

        # Q LOSS
        curr_q1 = self.model.q_net_1.forward(states, actions)
        curr_q2 = self.model.q_net_2.forward(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # UPDATE Q NETWORKS
        self.model.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.model.q1_optimizer.step()

        self.model.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.model.q2_optimizer.step()

        # UPDATE POLICY NETWORK
        new_actions, log_pi, _, _ = self._evaluate_action(states)
        min_q = torch.min(
            self.model.q_net_1.forward(states, new_actions),
            self.model.q_net_2.forward(states, new_actions),
        )
        policy_loss = (self.alpha * log_pi - min_q).mean()

        self.model.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.model.policy_optimizer.step()

        # UPDATE TARGET NETWORKS
        for target_param, param in zip(
            self.model.target_q_net_1.parameters(), self.model.q_net_1.parameters()
        ):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(
            self.model.target_q_net_2.parameters(), self.model.q_net_2.parameters()
        ):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # UPDATE TEMPERATURE
        alpha_loss = (self.model.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.model.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.model.alpha_optim.step()
        self.alpha = self.model.log_alpha.exp()

        self.update_step += 1

    def save_model(self, path: str):
        print("... saving model ...")
        torch.save(self.model.q_net_1.state_dict(), path + "_q_net1")
        torch.save(self.model.q_net_2.state_dict(), path + "_q_net2")

        torch.save(self.model.target_q_net_1.state_dict(), path + "_target_q_net1")
        torch.save(self.model.target_q_net_2.state_dict(), path + "_target_q_net2")

        torch.save(self.model.policy_net.state_dict(), path + "_policy_net")

        torch.save(self.alpha, path + "_alpha.pt")

    def load_model(self, path: str):
        print("... loading model ...")
        self.model.q_net_1.load_state_dict(torch.load(path + "_q_net1"))
        self.model.q_net_2.load_state_dict(torch.load(path + "_q_net2"))

        self.model.target_q_net_1.load_state_dict(torch.load(path + "_target_q_net1"))
        self.model.target_q_net_2.load_state_dict(torch.load(path + "_target_q_net2"))

        self.model.policy_net.load_state_dict(torch.load(path + "_policy_net"))

        self.alpha = torch.load(path + "_alpha.pt")

        self.model.q_net_1.eval()
        self.model.q_net_2.eval()
        self.model.target_q_net_1.eval()
        self.model.target_q_net_2.eval()
        self.model.policy_net.eval()

    def copy(self):
        copy = self.__class__(
            self.model.copy(),
            self.gamma,
            self.tau,
            self.reward_scaling,
            self.action_scaling,
            self.exploration_action_noise,
        )
        return copy

    def copy_shared_memory(self):
        copy = self.__class__(
            self.model.copy_shared_memory(),
            self.gamma,
            self.tau,
            self.reward_scaling,
            self.action_scaling,
            self.exploration_action_noise,
            device=self.device,
        )
        return copy
