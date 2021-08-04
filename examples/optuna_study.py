import stacierl
import stacierl.environment.tiltmaze as tiltmaze
import numpy as np
import torch
import csv
import optuna
import os
from lstm import sac_training

id = 0

name = "lstm"


def optuna_run(trial):
    cwd = os.getcwd()
    lr = trial.suggest_loguniform("lr", 5e-5, 1e-2)
    gamma = trial.suggest_float("gamma", 0.99, 0.9999)
    n_layers = trial.suggest_int("n_layers", 1, 2)
    n_nodes = trial.suggest_int("n_nodes", 128, 256)
    hidden_layers = [n_nodes for _ in range(n_layers)]
    n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2)
    n_lstm_nodes = trial.suggest_int("n_lstm_nodes", 64, 256)
    success, steps = sac_training(
        lr=lr,
        gamma=gamma,
        hidden_layers=hidden_layers,
        n_lstm_layer=n_lstm_layers,
        n_lstm_nodes=n_lstm_nodes,
        id=trial.number,
        training_steps=5e5,
        name=name,
        log_folder=cwd + "/optuna_results/",
    )
    with open(cwd + "/optuna_results/" + name + ".csv", "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([trial.number, success, trial.params, steps])
    return success


if __name__ == "__main__":

    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(optuna_run, n_trials=50)
