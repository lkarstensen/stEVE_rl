from typing import List, Optional, Tuple

import numpy as np
from torch.distributions.normal import Normal
from .sac import SAC
from .. import network
import torch.optim as optim
import torch


class SACsharedLSTM(SAC):
    def __init__(
        self,
        q_net_1: network.QNetworkLSTM,
        q_net_2: network.QNetworkLSTM,
        target_q_net_2: network.QNetworkLSTM,
        target_q_net_1: network.QNetworkLSTM,
        policy_net: network.GaussianPolicyLSTM,
        learning_rate: float = 0.0007525,
    ) -> None:
        super().__init__(
            q_net_1, q_net_2, target_q_net_1, target_q_net_2, policy_net, learning_rate
        )

        self._init_lstm()

    def _init_lstm(self):
        self.q_net_1.lstm = self.policy_net.lstm
        self.q_net_2.lstm = self.policy_net.lstm

    def get_action(
        self, flat_state: np.ndarray, hidden_state: Optional[torch.tensor] = None
    ) -> Tuple[np.ndarray, Optional[torch.tensor]]:
        flat_state = np.expand_dims(flat_state, 0)

        action, hidden_state_out = super().get_action(flat_state, hidden_state)

        action = action.squeeze(0)
        return action, hidden_state_out

    @property
    def initial_hidden_state(self) -> Optional[torch.Tensor]:
        hc = self.policy_net.initial_hidden_state
        h = hc[0].to(self.device)
        c = hc[1].to(self.device)
        return (h, c)
