import stacierl
import stacierl.environment.tiltmaze as tiltmaze
import numpy as np
import torch
import csv
import optuna
import os

id = 0

name = "shortest_training"


def sac_training(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.000766667,
    hidden_layers=[256, 256],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=5e5,
    consecutive_explore_episodes=1,
    steps_between_eval=1e4,
    eval_episodes=100,
    batch_size=64,
    heatup=1000,
    id=0,
):
    cwd = os.getcwd()
    success = 0.0
    env_factory = tiltmaze.LNK1(dt_step=2 / 3)
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
    algo = stacierl.algo.SAC(sac_model, gamma=gamma, device=device)
    replay_buffer = stacierl.replaybuffer.Vanilla(replay_buffer)
    agent = stacierl.agent.SingleAgent(algo, env, replay_buffer, consecutive_action_steps=1)

    if not os.path.isdir(cwd + "/optuna_results/"):
        os.mkdir(cwd + "/optuna_results/")

    logfile = cwd + "/optuna_results/" + name + "_" + str(id) + ".csv"
    with open(logfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(["lr", "gamma", "hidden_layers"])
        writer.writerow([lr, gamma, hidden_layers])
        writer.writerow(["Episodes", "Steps", "Reward", "Success"])

    next_eval_step_limt = steps_between_eval
    agent.heatup(steps=heatup)
    while agent.explore_step_counter < training_steps and success < 1.0:
        agent.explore(episodes=consecutive_explore_episodes)

        learn_steps = agent.explore_step_counter - agent.update_step_counter
        agent.update(learn_steps, batch_size)

        if agent.explore_step_counter > next_eval_step_limt:
            reward, success = agent.evaluate(episodes=eval_episodes)
            next_eval_step_limt += steps_between_eval

            print(f"Steps: {agent.explore_step_counter}, Reward: {reward}, Success: {success}")
            with open(logfile, "a+", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow(
                    [agent.explore_episode_counter, agent.explore_step_counter, reward, success]
                )

    return success, agent.explore_step_counter


def optuna_run(trial):
    cwd = os.getcwd()
    lr = trial.suggest_loguniform("lr", 3e-4, 1e-2)
    gamma = trial.suggest_float("gamma", 0.99, 0.9999)
    n_layers = trial.suggest_int("n_layers", 1, 2)
    n_nodes = trial.suggest_categorical("n_nodes", [128, 256])
    hidden_layers = [n_nodes for _ in range(n_layers)]
    success, steps = sac_training(lr=lr, gamma=gamma, hidden_layers=hidden_layers, id=trial.number)
    with open(cwd + "/optuna_results/" + name + ".csv", "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([trial.number, success, trial.params, steps])
    return steps


if __name__ == "__main__":

    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(optuna_run, n_trials=2)