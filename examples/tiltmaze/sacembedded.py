import stacierl
import tiltmaze
import numpy as np
import torch
from datetime import datetime
import csv
import os
import torch.multiprocessing as mp
from time import perf_counter
from stacierl.algo.sacmodel import Embedder
import math


def sac_training(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.000766667,
    hidden_layers=[256, 256],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=3e5,
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

    maze = tiltmaze.maze.RSS0(episodes_btw_geometry_change=math.inf, seed=1234)
    velocity_limits = (100, 0.5)
    physic = tiltmaze.physics.BallVelocity(
        velocity_limits=velocity_limits,
        action_scaling=velocity_limits,
        dt_step=1 / 7.5,
    )
    target = tiltmaze.target.CenterlineRandom(10)
    pathfinder = tiltmaze.pathfinder.NodesBFS()
    start = tiltmaze.start.CenterlineRandom()
    imaging = tiltmaze.imaging.ImagingDummy((500, 500))

    pos = tiltmaze.state.Position()
    pos = tiltmaze.state.wrapper.Normalize(pos)
    target_state = tiltmaze.state.Target()
    target_state = tiltmaze.state.wrapper.Normalize(target_state)
    state = tiltmaze.state.Combination([pos, target_state])

    target_reward = tiltmaze.reward.TargetReached(1.0)
    step_reward = tiltmaze.reward.Step(-0.005)
    path_length_reward = tiltmaze.reward.PathLengthDelta(0.001)
    reward = tiltmaze.reward.Combination([target_reward, step_reward, path_length_reward])

    done_target = tiltmaze.done.TargetReached()
    done_steps = tiltmaze.done.MaxSteps(100)
    done = tiltmaze.done.Combination([done_target, done_steps])

    success = tiltmaze.success.TargetReached()
    visu = tiltmaze.visualisation.VisualisationDummy()

    env = tiltmaze.Env(
        maze=maze,
        state=state,
        reward=reward,
        done=done,
        physic=physic,
        start=start,
        target=target,
        imaging=imaging,
        pathfinder=pathfinder,
        visualisation=visu,
        success=success,
    )

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
    replay_buffer = stacierl.replaybuffer.VanillaEpisode(replay_buffer, batch_size)
    agent = stacierl.agent.Parallel(
        algo,
        env,
        replay_buffer,
        n_agents=2,
        consecutive_action_steps=1,
        device=device,
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
        # n_agents=3,
        batch_size=6,
        device=torch.device("cuda"),
        training_steps=3e5,
        heatup=1e4,
        name="lnk3",
    )
