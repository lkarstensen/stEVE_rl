import stacierl
import stacierl.environment.eve as eve
import numpy as np
import torch
from datetime import datetime
import csv
import os

from stacierl.algo.sacmodel import Embedder


def sac_training(
    shutdown,
    worker_device=torch.device("cpu"),
    trainer_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.000766667,
    hidden_layers=[256, 256],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=1e6,
    consecutive_explore_steps=150,
    consecutive_explore_episodes=1,
    update_steps_per_exploration_step=1,
    steps_between_eval=1e5,
    eval_episodes=100,
    batch_size=8,
    heatup=10000,
    n_worker=20,
    n_trainer=4,
    log_folder: str = "",
    id=0,
    env_str: str = "lnk1",
    image_frequency=7.5,
    path_reward_factor=0.01,
):

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    success = 0.0
    env_str = env_str
    if env_str == "lnk1":
        env_factory = eve.LNK1(image_frequency, path_reward_factor)
    elif env_str == "lnk2":
        env_factory = eve.LNK2(image_frequency, path_reward_factor)
    elif env_str == "lnk3":
        env_factory = eve.LNK3(image_frequency, path_reward_factor)
    elif env_str == "lnk4":
        env_factory = eve.LNK4(image_frequency, path_reward_factor)
    env = env_factory.create_env()

    q_net_1 = stacierl.network.QNetwork(hidden_layers)
    q_net_2 = stacierl.network.QNetwork(hidden_layers)
    policy_net = stacierl.network.GaussianPolicy(hidden_layers, env.action_space)

    common_net = stacierl.network.LSTM(n_layer=1, n_nodes=128)

    sac_model = stacierl.algo.sacmodel.InputEmbedding(
        q1=q_net_1,
        q2=q_net_2,
        policy=policy_net,
        learning_rate=lr,
        obs_space=env.observation_space,
        action_space=env.action_space,
        q1_common_input_embedder=Embedder(common_net, True),
        q2_common_input_embedder=Embedder(common_net, False),
        policy_common_input_embedder=Embedder(common_net, False),
    )
    algo = stacierl.algo.SAC(sac_model, action_space=env.action_space, gamma=gamma)
    replay_buffer = stacierl.replaybuffer.VanillaEpisodeShared(replay_buffer, batch_size)
    agent = stacierl.agent.Synchron(
        n_worker=n_worker,
        n_trainer=n_trainer,
        algo=algo,
        env_factory=env_factory,
        replay_buffer=replay_buffer,
        worker_device=worker_device,
        trainer_device=trainer_device,
        share_trainer_model=True,
    )
    # agent = stacierl.agent.Parallel(
    #     n_agents=n_worker,
    #     algo=algo,
    #     env_factory=env_factory,
    #     replay_buffer=replay_buffer,
    #     device=worker_device,
    #     shared_model=True,
    # )

    logfile = f"{log_folder}/run_{id}.csv"
    id_2 = 0
    while True:
        if os.path.isfile(logfile):
            logfile = f"{log_folder}/run_{id}-{id_2}.csv"
            id_2 += 1
        else:
            break
    logfile = log_folder + "/run_" + str(id) + ".csv"
    with open(logfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(
            [
                "lr",
                "gamma",
                "hidden_layers",
                "image_frequency",
                "path_reward_factor",
                "environment",
                "batch_size",
            ]
        )
        writer.writerow(
            [lr, gamma, hidden_layers, image_frequency, path_reward_factor, env_str, batch_size]
        )
        writer.writerow(["Episodes", "Steps", "Reward", "Success"])

    next_eval_step_limt = steps_between_eval
    agent.heatup(steps=heatup)
    step_counter = agent.step_counter
    last_exporation_steps = step_counter.exploration
    while step_counter.exploration < training_steps and not shutdown.is_set():
        agent.explore(steps=consecutive_explore_steps)
        step_counter = agent.step_counter
        update_steps = int(
            (step_counter.exploration - last_exporation_steps) * update_steps_per_exploration_step
        )
        last_exporation_steps = step_counter.exploration
        agent.update(update_steps)

        if step_counter.exploration > next_eval_step_limt:
            reward, success = agent.evaluate(episodes=eval_episodes)
            reward = sum(reward) / len(reward)
            success = sum(success) / len(success)
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

    agent.close()

    return success, agent.step_counter.exploration


if __name__ == "__main__":
    import torch.multiprocessing as mp
    import argparse

    parser = argparse.ArgumentParser(description="Optuna Study.")
    parser.add_argument("logfolder", type=str, help="Folder to save logfiles")
    parser.add_argument("env", type=str, help="Environment to use")
    parser.add_argument("n_worker", type=int, help="Amount of Exploration Workers")
    parser.add_argument("n_trainer", type=int, help="Amount of NN Training Agents")
    parser.add_argument(
        "image_frequency",
        type=float,
        help="Frquency of the imaging system and therefore control frequency",
    )
    parser.add_argument(
        "path_reward_factor", type=float, help="Factor of the path_length_delta reward"
    )
    parser.add_argument("lr", type=float, help="Learning rate")
    parser.add_argument("name", type=str, help="an integer for the accumulator")
    args = parser.parse_args()
    name = args.name
    log_folder = args.logfolder
    n_worker = args.n_worker
    n_trainer = args.n_trainer
    env = args.env
    image_frequency = args.image_frequency
    path_reward_factor = args.path_reward_factor
    lr = args.lr
    mp.set_start_method("spawn", force=True)
    hidden_layers = [128, 128, 128]
    result = sac_training(
        lr=lr,
        gamma=0.999,
        hidden_layers=hidden_layers,
        log_folder=log_folder,
        id=name,
        n_worker=n_worker,
        n_trainer=n_trainer,
        env=env,
        image_frequency=image_frequency,
        path_reward_factor=path_reward_factor,
    )
