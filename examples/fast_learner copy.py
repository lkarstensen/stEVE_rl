import csv
import os
from torch import optim
import torch
import numpy as np

import eve_bench
import stacierl


def sac_training(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.003,
    hidden_layers=[255, 255, 255],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=1e6,
    consecutive_explore_steps=1,
    steps_between_eval=5e4,
    eval_episodes=100,
    batch_size=256,
    heatup=1e5,
    log_folder: str = os.getcwd() + "/fast_learner_example_results/",
    id_training=0,
    name="fast_learner",
    *args,
    **kwargs,
):

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    success = 0.0

    env = eve_bench.aorticarch.ArchVMR94(normalize_obs=True)

    obs_dict = env.observation_space.sample()
    obs_list = [obs.flatten() for obs in obs_dict.values()]
    obs_np = np.concatenate(obs_list)

    n_observations = obs_np.shape[0]
    n_actions = env.action_space.sample().flatten().shape[0]
    q_net_1 = stacierl.network.QNetwork(n_observations, n_actions, hidden_layers)
    q1_optimizer = optim.Adam(q_net_1.parameters(), 2e-4)
    q1_scheduler = optim.lr_scheduler.LinearLR(
        q1_optimizer, end_factor=5e-5, total_iters=1e5
    )

    q_net_2 = stacierl.network.QNetwork(n_observations, n_actions, hidden_layers)
    q2_optimizer = optim.Adam(q_net_2.parameters(), 2e-4)
    q2_scheduler = optim.lr_scheduler.LinearLR(
        q2_optimizer, end_factor=5e-5, total_iters=1e5
    )

    policy_net = stacierl.network.GaussianPolicy(
        n_observations, n_actions, hidden_layers
    )
    policy_optimizer = optim.Adam(policy_net.parameters(), 2e-4)
    policy_scheduler = optim.lr_scheduler.LinearLR(
        policy_optimizer, end_factor=5e-5, total_iters=1e5
    )

    sac_model = stacierl.algo.sacmodel.Vanilla(
        q1=q_net_1,
        q2=q_net_2,
        policy=policy_net,
        q1_optimizer=q1_optimizer,
        q2_optimizer=q2_optimizer,
        policy_optimizer=policy_optimizer,
        q1_scheduler=q1_scheduler,
        q2_scheduler=q2_scheduler,
        policy_scheduler=policy_scheduler,
        lr_alpha=lr,
    )
    algo = stacierl.algo.SAC(sac_model, n_actions=n_actions, gamma=gamma)
    replay_buffer = stacierl.replaybuffer.VanillaStepShared(replay_buffer, batch_size)
    agent = stacierl.agent.Synchron(
        algo,
        env,
        env,
        replay_buffer,
        consecutive_action_steps=1,
        normalize_actions=True,
        n_worker=6,
        trainer_device=device,
    )
    agent.save_config("/Users/lennartkarstensen/stacie/stacierl/test.yml")

    while True:
        logfile = log_folder + f"/{name}_{id_training}.csv"
        if os.path.isfile(logfile):
            id_training += 1
        else:
            break
    with open(logfile, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["lr", "gamma", "hidden_layers"])
        writer.writerow([lr, gamma, hidden_layers])
        writer.writerow(["Episodes", "Steps", "Reward", "Success"])
    # agent.load_checkpoint(log_folder, "checkpoint_10053.pt")
    next_eval_step_limt = steps_between_eval + agent.step_counter.exploration
    training_steps += agent.step_counter.exploration
    print("starting heatup")
    agent.heatup(steps=heatup, custom_action_low=[-10.0, -1.5])
    step_counter = agent.step_counter
    print("starting training loop")
    while step_counter.exploration < training_steps:
        agent.explore(steps=consecutive_explore_steps)
        step_counter = agent.step_counter
        update_steps = step_counter.exploration - step_counter.update
        agent.update(update_steps)

        if step_counter.exploration >= next_eval_step_limt:
            agent.save_checkpoint(
                os.path.join(log_folder, f"checkpoint_{step_counter.exploration}")
            )
            episodes = agent.evaluate(episodes=eval_episodes)
            rewards = [episode.episode_reward for episode in episodes]
            successes = [episode.infos[-1]["success"] for episode in episodes]
            reward = sum(rewards) / len(rewards)
            success = sum(successes) / len(successes)
            next_eval_step_limt += steps_between_eval

            print(
                f"Steps: {step_counter.exploration}, Reward: {reward}, Success: {success}"
            )
            with open(logfile, "a+", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(
                    [
                        agent.episode_counter.exploration,
                        step_counter.exploration,
                        reward,
                        success,
                    ]
                )

    return success, agent.step_counter.exploration


if __name__ == "__main__":
    result = sac_training(device=torch.device("mps"))
