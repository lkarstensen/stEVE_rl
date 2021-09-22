from torch.nn.functional import poisson_nll_loss
import stacierl
import stacierl.environment.tiltmaze as tiltmaze
import numpy as np
import torch
from datetime import datetime
import csv
import os
import torch.multiprocessing as mp
from time import perf_counter


def sac_training(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.000766667,
    hidden_layers=[256, 256],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=2e5,
    consecutive_explore_steps=1,
    steps_between_eval=1e4,
    eval_episodes=100,
    batch_size=8,
    heatup=1000,
    log_folder: str = "",
    n_agents=2,
    id=0,
    name="",
):

    start = perf_counter()
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    success = 0.0
    env_factory = tiltmaze.LNK2()
    env = env_factory.create_env()

    q_net_1 = stacierl.network.QNetwork(hidden_layers)
    q_net_2 = stacierl.network.QNetwork(hidden_layers)
    policy_net = stacierl.network.GaussianPolicy(hidden_layers, env.action_space)

    common_net = stacierl.network.LSTM(n_layer=1, n_nodes=128)
    common_embedder = stacierl.model.InputEmbedder("common", requires_grad=True)
    common_embedder_no_grad = stacierl.model.InputEmbedder("common", requires_grad=False)

    sac_model = stacierl.model.SACembedder(
        q1=q_net_1,
        q2=q_net_2,
        policy=policy_net,
        learning_rate=lr,
        obs_space=env.observation_space,
        action_space=env.action_space,
        embedding_networks={"common": common_net},
        q1_common_input_embedder=common_embedder,
        q2_common_input_embedder=common_embedder_no_grad,
        policy_common_input_embedder=common_embedder_no_grad,
    )
    algo = stacierl.algo.SAC(sac_model, action_space=env.action_space, gamma=gamma)
    replay_buffer = stacierl.replaybuffer.VanillaEpisode(replay_buffer, batch_size)
    agent = stacierl.agent.Single(
        algo, env, replay_buffer, consecutive_action_steps=1, device=device
    )

    while True:
        logfile = log_folder + f"/{name}_{id}.csv"
        if os.path.isfile(logfile):
            id += 1
        else:
            break
    with open(logfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["lr", "gamma", "hidden_layers"])
        writer.writerow([lr, gamma, hidden_layers])
        writer.writerow(["Episodes", "Steps", "Reward", "Success"])

    next_eval_step_limt = steps_between_eval
    agent.heatup(steps=heatup)
    step_counter = agent.step_counter
    while step_counter.exploration < training_steps:
        agent.explore(steps=consecutive_explore_steps)
        step_counter = agent.step_counter
        update_steps = step_counter.exploration - step_counter.update
        agent.update(update_steps)

        if step_counter.exploration >= next_eval_step_limt:
            reward, success = agent.evaluate(episodes=eval_episodes)
            next_eval_step_limt += steps_between_eval

            print(f"Steps: {step_counter.exploration}, Reward: {reward}, Success: {success}")
            with open(logfile, "a+", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(
                    [
                        agent.episode_counter.exploration,
                        step_counter.exploration,
                        reward,
                        success,
                    ]
                )
    duration = perf_counter() - start
    with open(logfile, "a+", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["Elapsed Time:"])
        writer.writerow([duration])
    agent.close()

    return success, agent.step_counter.exploration


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cwd = os.getcwd()
    log_folder = cwd + "/sacembedded_example_results/"
    result = sac_training(
        lr=0.0007,
        gamma=0.99,
        hidden_layers=[256, 256],
        log_folder=log_folder,
        n_agents=2,
        batch_size=8,
        device=torch.device("cuda"),
        training_steps=1e5,
        heatup=1e4,
        name="trial",
    )
