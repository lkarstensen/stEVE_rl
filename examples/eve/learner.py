from torch.nn.functional import poisson_nll_loss
import stacierl
import stacierl.environment.eve as eve
import numpy as np
import torch
from datetime import datetime
import csv
import os


def sac_training(
    worker_device=torch.device("cpu"),
    trainer_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.000766667,
    hidden_layers=[256, 256],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=1e6,
    consecutive_explore_episodes=50,
    update_steps_per_exploration_step=1,
    steps_between_eval=5e4,
    eval_episodes=100,
    batch_size=128,
    heatup=1000,
    n_worker=20,
    n_trainer=4,
    log_folder: str = "",
    id=0,
    env: str = "lnk1",
):

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    success = 0.0
    if env == "lnk1":
        env_factory = eve.LNK1()
    elif env == "lnk2":
        env_factory = eve.LNK2()
    env = env_factory.create_env()

    obs_dict_shape = env.observation_space.shape
    n_observations = 0
    for obs_shape in obs_dict_shape.values():
        n_observations += np.prod(obs_shape)
    n_actions = np.prod(env.action_space.shape)

    q_net_1 = stacierl.network.QNetwork(n_observations, n_actions, hidden_layers)
    q_net_2 = stacierl.network.QNetwork(n_observations, n_actions, hidden_layers)
    policy_net = stacierl.network.GaussianPolicy(n_observations, n_actions, hidden_layers)
    sac_model = stacierl.model.SAC(
        q_net_1=q_net_1,
        q_net_2=q_net_2,
        policy_net=policy_net,
        target_q_net_1=q_net_1.copy(),
        target_q_net_2=q_net_2.copy(),
        learning_rate=lr,
    )
    algo = stacierl.algo.SAC(sac_model, gamma=gamma)
    replay_buffer = stacierl.replaybuffer.VanillaShared(replay_buffer)
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

    logfile = log_folder + "/run_" + str(id) + ".csv"
    with open(logfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["lr", "gamma", "hidden_layers"])
        writer.writerow([lr, gamma, hidden_layers])
        writer.writerow(["Episodes", "Steps", "Reward", "Success"])

    next_eval_step_limt = steps_between_eval
    agent.heatup(steps=heatup)
    step_counter = agent.step_counter
    last_exporation_steps = step_counter.exploration
    while step_counter.exploration < training_steps:
        agent.explore(episodes=consecutive_explore_episodes)
        step_counter = agent.step_counter
        update_steps = int(
            (step_counter.exploration - last_exporation_steps) * update_steps_per_exploration_step
        )
        last_exporation_steps = step_counter.exploration
        agent.update(update_steps, batch_size)

        if step_counter.exploration > next_eval_step_limt:
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

    agent.close()

    return success, agent.step_counter.exploration


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    cwd = os.getcwd()
    log_folder = cwd + "/eve_learner_example_results/"
    hidden_layers = [128, 128, 128]

    result = sac_training(
        lr=0.0025,
        gamma=0.99,
        hidden_layers=hidden_layers,
        log_folder=log_folder,
    )
