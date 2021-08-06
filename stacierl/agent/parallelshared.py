from typing import List

from .parallel import Parallel, Algo, ReplayBuffer, EnvFactory, SingleAgentProcess
from math import ceil
import torch


class ParallelShared(Parallel):
    def __init__(
        self,
        n_agents: int,
        algo: Algo,
        env_factory: EnvFactory,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
    ) -> None:
        self.algo = algo
        self.env_factory = env_factory
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps
        self.n_agents = n_agents
        self.device = device

        self.agents: List[SingleAgentProcess] = []

        for i in range(n_agents):
            self.agents.append(
                SingleAgentProcess(
                    i,
                    algo.copy_shared_memory(),
                    self.env_factory.create_env(),
                    replay_buffer.copy(),
                    device,
                    consecutive_action_steps,
                )
            )

    def update(self, steps, batch_size):
        steps_per_agent = ceil(steps / self.n_agents)
        for agent in self.agents:
            agent.update(steps_per_agent, batch_size)
