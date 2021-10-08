from typing import Dict, Iterator, Optional, Tuple
import numpy as np
from torch.distributions.normal import Normal
from .vanilla import Vanilla, SACStateDicts
from ... import network
from ...network import NetworkDummy, Network
import torch.optim as optim
import torch
from dataclasses import dataclass
from copy import deepcopy

from ...util import ObservationSpace, ActionSpace


@dataclass
class Embedder:
    network: Network
    update: bool


@dataclass
class HydraNetwork:
    network: Network
    split_state_id: int
    requires_grad: bool


@dataclass
class SACEmbeddedStateDicts(SACStateDicts):
    q1: Dict[str, torch.Tensor]
    q2: Dict[str, torch.Tensor]
    target_q1: Dict[str, torch.Tensor]
    target_q2: Dict[str, torch.Tensor]
    policy: Dict[str, torch.Tensor]
    q1_common: Dict[str, torch.Tensor]
    q2_common: Dict[str, torch.Tensor]
    policy_common: Dict[str, torch.Tensor]
    q1_hydra: Dict[str, Dict[str, torch.Tensor]]
    q2_hydra: Dict[str, Dict[str, torch.Tensor]]
    policy_hydra: Dict[str, Dict[str, torch.Tensor]]

    def __iter__(self):
        iter_list = [
            self.q1,
            self.q2,
            self.target_q1,
            self.target_q2,
            self.policy,
            self.q1_common,
            self.q2_common,
            self.policy_common,
        ]
        iter_list += list(self.q1_hydra.values())
        iter_list += list(self.q2_hydra.values())
        iter_list += list(self.policy_hydra.values())
        return iter(iter_list)

    def copy(self):
        return SACEmbeddedStateDicts(
            deepcopy(self.q1),
            deepcopy(self.q2),
            deepcopy(self.target_q1),
            deepcopy(self.target_q2),
            deepcopy(self.policy),
            deepcopy(self.q1_common),
            deepcopy(self.q2_common),
            deepcopy(self.policy_common),
            deepcopy(self.q1_hydra),
            deepcopy(self.q2_hydra),
            deepcopy(self.policy_hydra),
        )


class InputEmbedding(Vanilla):
    def __init__(
        self,
        q1: network.QNetwork,
        q2: network.QNetwork,
        policy: network.GaussianPolicy,
        learning_rate: float,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        q1_common_input_embedder: Optional[Embedder] = None,
        q2_common_input_embedder: Optional[Embedder] = None,
        policy_common_input_embedder: Optional[Embedder] = None,
        q1_hydra_input_embedder: Optional[Dict[str, Embedder]] = None,
        q2_hydra_input_embedder: Optional[Dict[str, Embedder]] = None,
        policy_hydra_input_embedder: Optional[Dict[str, Embedder]] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.obs_space = obs_space
        self.action_space = action_space
        self.q1_hydra_input_embedders = q1_hydra_input_embedder or {}
        self.q2_hydra_input_embedders = q2_hydra_input_embedder or {}
        self.policy_hydra_input_embedders = policy_hydra_input_embedder or {}

        self.q1_common_input_embedder = q1_common_input_embedder
        self.q2_common_input_embedder = q2_common_input_embedder
        self.policy_common_input_embedder = policy_common_input_embedder

        self.q1 = q1
        self.q2 = q2
        self.target_q1 = q1.copy()
        self.target_q2 = q2.copy()
        self.policy = policy
        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.dict_to_flat_np_map = self.obs_space.dict_to_flat_np_map
        slices = [[ids, obs] for obs, ids in self.dict_to_flat_np_map.items()]
        slices = sorted(slices)
        self._dsplit_sections = [obs[0][0] for obs in slices if obs[0][0] != 0]
        self._state_key_to_split_state_index = {slices[i][1]: i for i in range(len(slices))}

        self._q1_hydra_networks = self._init_hydra_network(self.q1_hydra_input_embedders)
        self._q2_hydra_networks = self._init_hydra_network(self.q2_hydra_input_embedders)
        self._policy_hydra_networks = self._init_hydra_network(self.policy_hydra_input_embedders)

        self.q1_common_input_embedder = self._init_common_embedder(
            self.q1_common_input_embedder, self._q1_hydra_networks
        )
        self.q2_common_input_embedder = self._init_common_embedder(
            self.q2_common_input_embedder, self._q2_hydra_networks
        )
        self.policy_common_input_embedder = self._init_common_embedder(
            self.policy_common_input_embedder, self._policy_hydra_networks
        )
        n_actions = 1
        for dim in self.action_space.shape:
            n_actions *= dim
        self.q1.set_input(self.q1_common_input_embedder.network.n_outputs, n_actions)
        self.q2.set_input(self.q2_common_input_embedder.network.n_outputs, n_actions)
        self.target_q1.set_input(self.q1_common_input_embedder.network.n_outputs, n_actions)
        self.target_q2.set_input(self.q2_common_input_embedder.network.n_outputs, n_actions)
        self.policy.set_input(self.policy_common_input_embedder.network.n_outputs)
        self._init_optimizer()

    def _init_hydra_network(
        self,
        hydra_input_embedder: Dict[str, Embedder],
    ) -> Dict[str, HydraNetwork]:
        hydra_networks = {}
        for state_key, input_embedder in hydra_input_embedder.items():
            ids = self.dict_to_flat_np_map[state_key]
            n_observations = ids[1] - ids[0]
            network = input_embedder.network
            if network.input_is_set:
                if network.n_inputs != n_observations:
                    raise RuntimeError(
                        f"Input Embedder assignment seems to be wrong. Input Embedders always need the same number of inputs. Network {network} is wrongly assigned for state {state_key}."
                    )
            else:
                network.set_input(n_observations)

            hydra_networks.update(
                {
                    state_key: HydraNetwork(
                        network,
                        self._state_key_to_split_state_index[state_key],
                        input_embedder.update,
                    )
                }
            )

        for state_key in self.dict_to_flat_np_map.keys():
            if not state_key in hydra_networks.keys():
                dummy = NetworkDummy()
                hydra_networks.update({state_key: HydraNetwork(dummy, None, False)})
                ids = self.dict_to_flat_np_map[state_key]
                n_observations = ids[1] - ids[0]
                dummy.set_input(n_observations)

        return hydra_networks

    def _init_common_embedder(
        self, common_input_embedder: Embedder, hydra_networks: Dict[str, HydraNetwork]
    ):
        to_delete = []
        hydra_out = 0
        for state_key, state_network in hydra_networks.items():
            hydra_out += state_network.network.n_outputs
            if isinstance(state_network.network, NetworkDummy):
                to_delete.append(state_key)

        for state in to_delete:
            hydra_networks.pop(state)

        if common_input_embedder is None:
            network = NetworkDummy()
            network.set_input(hydra_out)
            update = False
        else:
            network = common_input_embedder.network
            update = common_input_embedder.update
            if network.input_is_set:
                if network.n_inputs != hydra_out:
                    raise RuntimeError(
                        f"Input Embedder assignment seems to be wrong. Input Embedders always need the same number of inputs. Common Embedder Network {network} is wrongly assigned for q1_common_embedder."
                    )
            else:
                network.set_input(hydra_out)

        return Embedder(network, update)

    def _init_optimizer(self):
        self.q1_optimizers = self._init_leg_optimizer(
            self.q1_hydra_input_embedders,
            self.q1_common_input_embedder,
            self.q1,
        )
        self.q2_optimizers = self._init_leg_optimizer(
            self.q2_hydra_input_embedders,
            self.q2_common_input_embedder,
            self.q2,
        )
        self.policy_optimizers = self._init_leg_optimizer(
            self.policy_hydra_input_embedders,
            self.policy_common_input_embedder,
            self.policy,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

    def _init_leg_optimizer(
        self,
        hydra_embedder: Dict[str, Embedder],
        common_embedder: Embedder,
        main_net: Network,
    ):
        optimizers = [optim.Adam(main_net.parameters(), lr=self.learning_rate)]
        for embedder in hydra_embedder.values():
            if embedder.update:
                optimizer = optim.Adam(embedder.network.parameters(), lr=self.learning_rate)
                optimizers.append(optimizer)
            else:
                continue

        if isinstance(common_embedder.network, NetworkDummy):
            common_embedder = None
        else:
            if common_embedder.update:
                optimizer = optim.Adam(common_embedder.network.parameters(), lr=self.learning_rate)
                optimizers.append(optimizer)
        return optimizers

    def get_play_action(self, flat_state: np.ndarray = None) -> np.ndarray:
        with torch.no_grad():
            flat_state = torch.from_numpy(flat_state).unsqueeze(0).unsqueeze(0)
            flat_state = flat_state.to(self.device)
            embedded_state = self._get_embedded_state(
                flat_state,
                self._policy_hydra_networks,
                self.policy_common_input_embedder,
                use_hidden_state=True,
            )

            mean, log_std = self.policy.forward(embedded_state, use_hidden_state=True)
            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
            return action

    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._q1_hydra_networks,
            self.q1_common_input_embedder,
            use_hidden_state=False,
        )
        q1 = self.q1(embedded_state, action_batch, use_hidden_state=False)
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._q2_hydra_networks,
            self.q2_common_input_embedder,
            use_hidden_state=False,
        )
        q2 = self.q2(embedded_state, action_batch, use_hidden_state=False)
        return q1, q2

    def get_target_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # with torch.no_grad():
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._q1_hydra_networks,
            self.q1_common_input_embedder,
            use_hidden_state=False,
        )
        q1 = self.target_q1(embedded_state, action_batch, use_hidden_state=False)
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._q2_hydra_networks,
            self.q2_common_input_embedder,
            use_hidden_state=False,
        )
        q2 = self.target_q2(embedded_state, action_batch, use_hidden_state=False)
        return q1, q2

    # epsilon makes sure that log(0) does not occur
    def get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._policy_hydra_networks,
            self.policy_common_input_embedder,
            use_hidden_state=False,
        )
        mean_batch, log_std = self.policy.forward(embedded_state, use_hidden_state=False)

        std_batch = log_std.exp()
        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)
        return action_batch, log_pi_batch

    def _get_embedded_state(
        self,
        state_batch: torch.Tensor,
        hydra_nets: Dict[str, HydraNetwork],
        common_embedder: Embedder,
        use_hidden_state: bool,
    ):

        if hydra_nets:

            sliced_state = list(state_batch.dsplit(self._dsplit_sections))
            for hydra_network in hydra_nets.values():
                if hydra_network.requires_grad:
                    reduced_output = hydra_network.network.forward(
                        sliced_state[hydra_network.split_state_id],
                        use_hidden_state=use_hidden_state,
                    )
                else:
                    with torch.no_grad():
                        reduced_output = hydra_network.network.forward(
                            sliced_state[hydra_network.split_state_id],
                            use_hidden_state=use_hidden_state,
                        )
                sliced_state[hydra_network.split_state_id] = reduced_output

            hydra_state = torch.dstack(sliced_state)
        else:
            hydra_state = state_batch

        if common_embedder is None:
            embedded_state = hydra_state
        else:
            if common_embedder.update:
                embedded_state = common_embedder.network.forward(
                    hydra_state, use_hidden_state=use_hidden_state
                )
            else:
                with torch.no_grad():
                    embedded_state = common_embedder.network.forward(
                        hydra_state, use_hidden_state=use_hidden_state
                    )
        return embedded_state

    def q1_update_zero_grad(self):
        for optimizer in self.q1_optimizers:
            optimizer.zero_grad()

    def q2_update_zero_grad(self):
        for optimizer in self.q2_optimizers:
            optimizer.zero_grad()

    def policy_update_zero_grad(self):
        for optimizer in self.policy_optimizers:
            optimizer.zero_grad()

    def alpha_update_zero_grad(self):
        self.alpha_optimizer.zero_grad()

    def q1_update_step(self):
        for optimizer in self.q1_optimizers:
            optimizer.step()

    def q2_update_step(self):
        for optimizer in self.q2_optimizers:
            optimizer.step()

    def policy_update_step(self):
        for optimizer in self.policy_optimizers:
            optimizer.step()

    def alpha_update_step(self):
        self.alpha_optimizer.step()

    def alpha_update(self, loss: torch.Tensor):
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

    def to(self, device: torch.device):
        self.device = device
        self.q1.to(device)
        self.q2.to(device)
        self.target_q1.to(device)
        self.target_q2.to(device)
        self.policy.to(device)
        self.q1_common_input_embedder.network.to(device)
        self.q2_common_input_embedder.network.to(device)
        self.policy_common_input_embedder.network.to(device)
        for net in self._q1_hydra_networks.values():
            net.network.to(device)
        for net in self._q2_hydra_networks.values():
            net.network.to(device)
        for net in self._policy_hydra_networks.values():
            net.network.to(device)

        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self._init_optimizer()

    def update_target_q(self, tau):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def copy(self):
        q1_common = Embedder(
            self.q1_common_input_embedder.network.copy(), self.q1_common_input_embedder.update
        )
        if self.q2_common_input_embedder.network == self.q1_common_input_embedder.network:
            q2_common_net = q1_common.network
        else:
            q2_common_net = self.q2_common_input_embedder.network.copy()
        q2_common = Embedder(q2_common_net, self.q2_common_input_embedder.update)

        if self.policy_common_input_embedder.network == self.q1_common_input_embedder.network:
            policy_common_net = q1_common.network
        elif self.policy_common_input_embedder.network == self.q2_common_input_embedder.network:
            policy_common_net = q2_common.network
        else:
            policy_common_net = self.policy_common_input_embedder.network.copy()
        policy_common = Embedder(policy_common_net, self.policy_common_input_embedder.update)
        q1_hydra = {}
        q2_hydra = {}
        policy_hydra = {}
        for key in self.dict_to_flat_np_map.keys():
            if key in self.q1_hydra_input_embedders.keys():
                q1 = Embedder(
                    self.q1_hydra_input_embedders[key].network.copy(),
                    self.q1_hydra_input_embedders[key].update,
                )
                q1_hydra.update({key: q1})
            if key in self.q2_hydra_input_embedders.keys():
                if (
                    self.q2_hydra_input_embedders[key].network
                    == self.q1_hydra_input_embedders[key].network
                ):
                    q2_net = q1.network
                else:
                    q2_net = self.q2_hydra_input_embedders[key].network.copy()
                q2 = Embedder(q2_net, self.q2_hydra_input_embedders[key].update)
                q2_hydra.update({key: q2})
            if key in self.policy_hydra_input_embedders.keys():
                if (
                    self.policy_hydra_input_embedders[key].network
                    == self.q1_hydra_input_embedders[key].network
                ):
                    policy_net = q1.network
                elif (
                    self.policy_hydra_input_embedders[key].network
                    == self.q2_hydra_input_embedders[key].network
                ):
                    policy_net = q2.network
                else:
                    policy_net = self.policy_hydra_input_embedders[key].network.copy()
                policy = Embedder(policy_net, self.policy_hydra_input_embedders[key].update)
                policy_hydra.update({key: policy})

        copy = self.__class__(
            self.q1.copy(),
            self.q2.copy(),
            self.policy.copy(),
            self.learning_rate,
            self.obs_space,
            self.action_space,
            q1_common,
            q2_common,
            policy_common,
            q1_hydra,
            q2_hydra,
            policy_hydra,
        )

        return copy

    def copy_shared_memory(self):
        copy = self.copy()

        self.q1.share_memory()
        self.q2.share_memory()
        self.target_q1.share_memory()
        self.target_q2.share_memory()
        self.policy.share_memory()
        self.q1_common_input_embedder.network.share_memory()
        self.q2_common_input_embedder.network.share_memory()
        self.policy_common_input_embedder.network.share_memory()

        copy.q1 = self.q1
        copy.q2 = self.q2
        copy.target_q1 = self.target_q1
        copy.target_q2 = self.target_q2
        copy.policy = self.policy
        copy.q1_common_input_embedder.network = self.q1_common_input_embedder.network
        copy.q2_common_input_embedder.network = self.q2_common_input_embedder.network
        self.policy_common_input_embedder.network = self.policy_common_input_embedder.network

        for state_key, hydra_net in self._q1_hydra_networks.items():
            copy._q1_hydra_networks[state_key].network = hydra_net.network
            copy.q1_hydra_input_embedders[state_key].network = self.q1_hydra_input_embedders[
                state_key
            ].network
        for state_key, hydra_net in self._q2_hydra_networks.items():
            copy._q2_hydra_networks[state_key].network = hydra_net.network
            copy.q2_hydra_input_embedders[state_key].network = self.q2_hydra_input_embedders[
                state_key
            ].network
        for state_key, hydra_net in self._policy_hydra_networks.items():
            copy._policy_hydra_networks[state_key].network = hydra_net.network
            copy.policy_hydra_input_embedders[
                state_key
            ].network = self.policy_hydra_input_embedders[state_key].network

        return copy

    def load_state_dicts(self, state_dicts: SACEmbeddedStateDicts):
        self.q1.load_state_dict(state_dicts.q1)
        self.q2.load_state_dict(state_dicts.q2)
        self.target_q1.load_state_dict(state_dicts.target_q1)
        self.target_q2.load_state_dict(state_dicts.target_q2)
        self.q1_common_input_embedder.network.load_state_dict(state_dicts.q1_common)
        self.q2_common_input_embedder.network.load_state_dict(state_dicts.q2_common)
        self.policy_common_input_embedder.network.load_state_dict(state_dicts.policy_common)
        for state, hydra_net in self._q1_hydra_networks.items():
            hydra_net.network.load_state_dict(state_dicts.q1_hydra[state])
        for state, hydra_net in self._q2_hydra_networks.items():
            hydra_net.network.load_state_dict(state_dicts.q2_hydra[state])
        for state, hydra_net in self._policy_hydra_networks.items():
            hydra_net.network.load_state_dict(state_dicts.policy_hydra[state])

    @property
    def state_dicts(self) -> SACEmbeddedStateDicts:
        q1_hydra = {
            state: hydra_net.network.state_dict()
            for state, hydra_net in self._q1_hydra_networks.items()
        }
        q2_hydra = {
            state: hydra_net.network.state_dict()
            for state, hydra_net in self._q2_hydra_networks.items()
        }
        policy_hydra = {
            state: hydra_net.network.state_dict()
            for state, hydra_net in self._policy_hydra_networks.items()
        }

        state_dicts = SACEmbeddedStateDicts(
            self.q1.state_dict(),
            self.q2.state_dict(),
            self.target_q1.state_dict(),
            self.target_q2.state_dict(),
            self.policy.state_dict(),
            self.q1_common_input_embedder.network.state_dict(),
            self.q2_common_input_embedder.network.state_dict(),
            self.policy_common_input_embedder.network.state_dict(),
            q1_hydra,
            q2_hydra,
            policy_hydra,
        )
        return state_dicts

    def reset(self) -> None:
        for net in self:
            net.reset()

    def __iter__(self) -> Iterator[Network]:
        nets = [
            self.q1,
            self.q2,
            self.target_q1,
            self.target_q2,
            self.policy,
            self.q1_common_input_embedder.network,
            self.q2_common_input_embedder.network,
            self.policy_common_input_embedder.network,
        ]
        q1_hydra = [hydra_net.network for hydra_net in self._q1_hydra_networks.values()]
        q2_hydra = [hydra_net.network for hydra_net in self._q2_hydra_networks.values()]
        policy_hydra = [hydra_net.network for hydra_net in self._policy_hydra_networks.values()]
        return iter(nets + q1_hydra + q2_hydra + policy_hydra)
