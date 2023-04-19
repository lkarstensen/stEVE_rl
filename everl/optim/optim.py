from abc import ABC, abstractmethod
from typing import Dict, List, Union
from torch import optim
from ..network import Network
from ..util import EveRLObject


class Optimizer(EveRLObject, ABC):
    @abstractmethod
    def __init__(
        self, networks: Union[Network, List[Network], List[Dict]], *args, **kwargs
    ) -> None:
        self.networks = networks

    def _networks_to_params(self, networks):
        if isinstance(networks, Network):
            networks = [networks]

        if isinstance(networks[0], Network):
            params = self._networks_list_to_params(networks)
        else:
            params = self._networks_groups_to_params_groups(networks)
        return params

    def _networks_list_to_params(self, networks: List[Network]):
        params = []
        for network in networks:
            params += network.parameters()
        return params

    def _networks_groups_to_params_groups(self, network_groups: List[Dict]):
        params = []
        for network_group in network_groups:
            params_group = network_group.copy()
            networks = network_group.pop("networks")
            params_group["params"] = self._networks_list_to_params(networks)

            params.append(params_group)

        return params


class Adam(optim.Adam, Optimizer):
    def __init__(
        self,
        networks: Union[Network, List[Network], List[Dict]],
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None
    ) -> None:
        self.networks = networks

        optim.Adam.__init__(
            self,
            self._networks_to_params(networks),
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        for key, value in self.defaults.items():
            setattr(self, key, value)
