import csv
import optuna
import os
from lstm import sac_training
import torch.multiprocessing as mp
import argparse


def optuna_run(trial):
    cwd = os.getcwd()
    if not os.path.isdir(cwd + "/optuna_results/"):
        os.mkdir(cwd + "/optuna_results/")
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    gamma = trial.suggest_float("gamma", 0.98, 0.9999)
    # n_layers = trial.suggest_int("n_layers", 1, 3)
    # n_nodes = trial.suggest_int("n_nodes", 32, 256)
    # hidden_layers = [n_nodes for _ in range(n_layers)]
    n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2)
    n_lstm_nodes = trial.suggest_int("n_lstm_nodes", 32, 256)
    success, steps = sac_training(
        lr=lr,
        gamma=gamma,
        # hidden_layers=hidden_layers,
        n_lstm_layer=n_lstm_layers,
        n_lstm_nodes=n_lstm_nodes,
        id=trial.number,
        name=name,
        log_folder=cwd + "/optuna_results/" + name,
    )
    with open(cwd + "/optuna_results/" + name + ".csv", "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([trial.number, success, trial.params, steps])
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Study.")
    parser.add_argument("name", type=str, help="an integer for the accumulator")
    parser.add_argument("n_trials", type=int, help="number of study trials")
    args = parser.parse_args()
    name = args.name
    n_trials = args.n_trials
    mp.set_start_method("spawn", force=True)
    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(optuna_run, n_trials=n_trials)
