from torch.nn.functional import poisson_nll_loss
import stacierl
import stacierl.environment.tiltmaze as tiltmaze
import numpy as np
import torch
from time import time
import csv
import os
import torch.multiprocessing as mp


def sac_training(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.005857455980764544,
    hidden_layers=[128, 128],
    n_lstm_nodes: int = 128,
    n_lstm_layer: int = 1,
    gamma=0.990019014056533,
    replay_buffer=1e4,
    training_steps=2e5,
    consecutive_explore_episodes=1,
    steps_between_eval=1e4,
    eval_episodes=100,
    batch_size=10,
    heatup=1000,
    sequence_length=10,
    log_folder: str = "",
    id: int = 0,
    name="lstm",
    n_agents=4,
):

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    success = 0.0
    env_factory = tiltmaze.LNK1(dt_step=2 / 3)
    env = env_factory.create_env()

    obs_dict_shape = env.observation_space.shape
    n_observations = 0
    for obs_shape in obs_dict_shape.values():
        n_observations += np.prod(obs_shape)
    n_actions = np.prod(env.action_space.shape)

    q_net_1 = stacierl.network.QNetworkLSTM(
        n_observations, n_actions, hidden_layers, n_lstm_nodes, n_lstm_layer
    )
    q_net_2 = stacierl.network.QNetworkLSTM(
        n_observations, n_actions, hidden_layers, n_lstm_nodes, n_lstm_layer
    )
    policy_net = stacierl.network.GaussianPolicyLSTM(
        n_observations, n_actions, hidden_layers, n_lstm_nodes, n_lstm_layer
    )
    sac_model = stacierl.model.SACsharedLSTM(
        q_net_1=q_net_1,
        q_net_2=q_net_2,
        policy_net=policy_net,
        target_q_net_1=q_net_1.copy(),
        target_q_net_2=q_net_2.copy(),
        learning_rate=lr,
    )
    algo = stacierl.algo.SAC(sac_model, gamma=gamma, device=device)
    replay_buffer = stacierl.replaybuffer.VanillaLSTM(
        replay_buffer, sequence_length=sequence_length
    )
    agent = stacierl.agent.ParallelAgent(
        n_agents, algo, env_factory, replay_buffer, consecutive_action_steps=1
    )

    logfile = log_folder + name + "_" + str(id) + ".csv"
    with open(logfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["lr", "gamma", "hidden_layers", "n_lstm_layers", "n_lstm_nodes"])
        writer.writerow([lr, gamma, hidden_layers, n_lstm_layer, n_lstm_nodes])
        writer.writerow(["Episodes", "Steps", "Reward", "Success"])

    next_eval_step_limt = steps_between_eval
    agent.heatup(steps=heatup)
    while agent.explore_step_counter < training_steps:
        agent.explore(episodes=consecutive_explore_episodes)

        update_steps = agent.explore_step_counter - agent.update_step_counter
        agent.update(update_steps, batch_size)

        if agent.explore_step_counter > next_eval_step_limt:
            reward, success = agent.evaluate(episodes=eval_episodes)
            next_eval_step_limt += steps_between_eval

            print(f"Steps: {agent.explore_step_counter}, Reward: {reward}, Success: {success}")
            with open(logfile, "a+", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(
                    [agent.explore_episode_counter, agent.explore_step_counter, reward, success]
                )
    agent.close()

    return success, agent.explore_step_counter


if __name__ == "__main__":
    time_start = time()
    mp.set_start_method("spawn")
    cwd = os.getcwd()
    log_folder = cwd + "/lstm_example_results/"
    result = sac_training(
        lr=0.005857455980764544,
        gamma=0.990019014056533,
        hidden_layers=[128, 128],
        n_lstm_nodes=128,
        n_lstm_layer=2,
        log_folder=log_folder,
        training_steps=1e5,
    )
    duration = time() - time_start
    print(f"Duration: {duration}")
