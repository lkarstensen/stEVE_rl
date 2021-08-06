import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .qnetworklstm import QNetworkLSTM


class QNetworkLSTM2(QNetworkLSTM):
    def forward(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        hidden_state_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            lstm_output, hidden_out = self.lstm(state_batch, hidden_state_batch)

        input = torch.cat([lstm_output, action_batch], dim=-1)
        for i in range(len(self.layers) - 1):
            output = self.layers[i](input)
            output = F.relu(output)
            input = output

        # output without relu
        q_value_batch = self.layers[-1](output)

        return q_value_batch, hidden_out
