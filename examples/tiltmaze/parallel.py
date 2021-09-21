from torch.nn.functional import poisson_nll_loss
import stacierl
import stacierl.environment.tiltmaze as tiltmaze
import numpy as np
import torch
from datetime import datetime
import csv
import os
import torch.multiprocessing as mp


def sac_training(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.000766667,
    hidden_layers=[256, 256],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=2e5,
    consecutive_explore_episodes=1,
    steps_between_eval=2e4,
    eval_episodes=100,
    batch_size=64,
    heatup=5000,
    log_folder: str = "",
    n_agents=3,
    id=0,
    name="",
):

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    success = 0.0
    env_factory = tiltmaze.LNK1(dt_step=2 / 3)
    env = env_factory.create_env()

    q_net_1 = stacierl.network.QNetwork(hidden_layers)
    q_net_2 = stacierl.network.QNetwork(hidden_layers)
    policy_net = stacierl.network.GaussianPolicy(hidden_layers, action_space=env.action_space)
    sac_model = stacierl.model.SAC(
        q1=q_net_1,
        q2=q_net_2,
        policy=policy_net,
        learning_rate=lr,
        obs_space=env.observation_space,
        action_space=env.action_space,
    )
    algo = stacierl.algo.SAC(sac_model, action_space=env.action_space, gamma=gamma)
    replay_buffer = stacierl.replaybuffer.Vanilla(replay_buffer, batch_size)
    agent = stacierl.agent.Parallel(
        n_agents,
        algo,
        env_factory,
        replay_buffer,
        device=device,
        consecutive_action_steps=1,
        shared_model=False,
    )

    logfile = log_folder + f"/{name}_{id}.csv"
    with open(logfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["lr", "gamma", "hidden_layers"])
        writer.writerow([lr, gamma, hidden_layers])
        writer.writerow(["Episodes", "Steps", "Reward", "Success"])

    next_eval_step_limt = steps_between_eval
    agent.heatup(steps=heatup)
    step_counter = agent.step_counter
    while step_counter.exploration < training_steps:
        agent.explore(episodes=consecutive_explore_episodes)
        step_counter = agent.step_counter
        update_steps = step_counter.exploration - step_counter.update
        agent.update(update_steps)

        if step_counter.exploration > next_eval_step_limt:
            reward, success = agent.evaluate(episodes=eval_episodes)
            next_eval_step_limt += steps_between_eval

            print(f"Steps: {step_counter.exploration}, Reward: {reward}, Success: {success}")
            with open(logfile, "a+", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(
                    [agent.episode_counter.exploration, step_counter.exploration, reward, success]
                )

    agent.close()
    return (success, step_counter)  # agent.explore_step_counter


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cwd = os.getcwd()
    log_folder = cwd + "/parallel_example_results/"
    result = sac_training(
        lr=0.002717137468421826,
        gamma=0.9867414187511384,
        hidden_layers=[61, 61],
        log_folder=log_folder,
        n_agents=3,
        batch_size=243,
        device=torch.device("cuda"),
        training_steps=3e5,
        heatup=1e4,
    )
