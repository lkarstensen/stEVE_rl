from typing import List
from .model import Model
from .. import network
import torch.optim as optim
import torch


class SAC(Model):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_layers: List[int],
        learning_rate: float = 0.0007525,
    ) -> None:
        self.learning_rate = learning_rate
        self.q_net_1 = network.QNetwork(n_observations, n_actions, hidden_layers=hidden_layers)
        self.q_net_2 = network.QNetwork(n_observations, n_actions, hidden_layers=hidden_layers)
        self.policy_net = network.GaussianPolicy(
            n_observations, n_actions, hidden_layers=hidden_layers
        )
        self.q1_optimizer = optim.Adam(self.q_net_1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q_net_2.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate / 2)

        self.target_q_net_1 = self.q_net_1.copy()
        self.target_q_net_2 = self.q_net_2.copy()

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=learning_rate)

    def to(self, device: torch.device):
        self.device = device
        self.q_net_1.to(device)
        self.q_net_2.to(device)
        self.target_q_net_1.to(device)
        self.target_q_net_2.to(device)
        self.policy_net.to(device)
        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        # torch.ones(1, requires_grad=True, device=device.type)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.learning_rate)
