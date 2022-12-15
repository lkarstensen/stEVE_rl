from typing import Tuple
import torch.optim as optim
from ..network import Network


class Optimizer(optim.Optimizer):
    def __init__(self, network: Network, default: dict) -> None:
        super().__init__(network.parameters(), default)
        self.network = network


class Adam(optim.Adam):
    def __init__(
        self,
        network: Network,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(network.parameters(), lr, betas, eps, weight_decay, amsgrad)
        self.network = network
