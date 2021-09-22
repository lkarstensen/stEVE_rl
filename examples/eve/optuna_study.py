import logging
import csv
import optuna
import os
from learner import sac_training
import torch.multiprocessing as mp
import argparse
import signal
import multiprocessing as mp

shutdown = mp.Event()


def sigterm_callback(nr, frame):
    print("received sigterm")
    shutdown.set()


def optuna_run(trial):
    if shutdown.is_set():
        return None
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    gamma = trial.suggest_float("gamma", 0.99, 0.999)
    n_layers = trial.suggest_int("n_layers", 2, 3)
    n_nodes = trial.suggest_int("n_nodes", 64, 256)
    hidden_layers = [n_nodes for _ in range(n_layers)]
    try:
        success, steps = sac_training(
            shutdown=shutdown,
            lr=lr,
            gamma=gamma,
            hidden_layers=hidden_layers,
            id=trial.number,
            log_folder=log_folder,
            n_worker=n_worker,
            n_trainer=n_trainer,
            env_str=env,
            image_frequency=image_frequency,
            path_reward_factor=path_reward_factor,
        )
    except ValueError:
        success = -1
    with open(log_folder + "/results.csv", "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(
            [trial.number, success, trial.params, steps, image_frequency, path_reward_factor, env]
        )
    return success


if __name__ == "__main__":

    signal.signal(signal.SIGTERM, sigterm_callback)
    parser = argparse.ArgumentParser(description="Optuna Study.")

    parser.add_argument("logfolder", type=str, help="Folder to save logfiles")
    parser.add_argument("n_trials", type=int, help="number of study trials")
    parser.add_argument("n_worker", type=int, help="Amount of Exploration Workers")
    parser.add_argument("n_trainer", type=int, help="Amount of NN Training Agents")
    parser.add_argument("env", type=str, help="Environment to use")
    parser.add_argument(
        "image_frequency",
        type=float,
        help="Frquency of the imaging system and therefore control frequency",
    )
    parser.add_argument(
        "path_reward_factor", type=float, help="Factor of the path_length_delta reward"
    )

    parser.add_argument("name", type=str, help="an integer for the accumulator")
    args = parser.parse_args()
    name = args.name
    n_trials = args.n_trials
    log_folder = args.logfolder
    n_worker = args.n_worker
    n_trainer = args.n_trainer
    env = args.env
    image_frequency = args.image_frequency
    path_reward_factor = args.path_reward_factor
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    log_folder = f"{log_folder}/{name}/"

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    if not os.path.isdir(log_folder + "/logging"):
        os.mkdir(log_folder + "/logging")
    logging.basicConfig(
        filename=f"{log_folder}/logging/optuna_study.log",
        level=logging.INFO,
        format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    )
    logging.info("logging initialized")
    mp.set_start_method("spawn", force=True)
    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(optuna_run, n_trials=n_trials)
