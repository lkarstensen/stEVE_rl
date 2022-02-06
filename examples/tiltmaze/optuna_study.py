import csv
import optuna
import os
from fast_learner import sac_training
import torch.multiprocessing as mp
import argparse


def optuna_run(trial):
    cwd = os.getcwd()
    if not os.path.isdir(cwd + "/optuna_results/"):
        os.mkdir(cwd + "/optuna_results/")
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    gamma = trial.suggest_loguniform("gamma", 0.98, 0.9999)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_nodes = trial.suggest_int("n_nodes", 64, 256)
    batch_size = trial.suggest_int("batch_size", 64, 256)
    hidden_layers = [n_nodes for _ in range(n_layers)]
    try:
        success, steps = sac_training(
            lr=lr,
            gamma=gamma,
            hidden_layers=hidden_layers,
            id=trial.number,
            name=name,
            log_folder=f"{cwd}/optuna_results/{name}",
            training_steps=1.5e5,
            heatup=1e5,
            batch_size=batch_size,
            n_agents=1,
        )
    except ValueError:
        success = -1
        steps = -1
    with open(f"{cwd}/optuna_results/{name}/results.csv", "a+") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow([trial.number, success, trial.params, steps])
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Study.")
    parser.add_argument("name", type=str, help="name of the trial", default="tiltmaze_trial")
    parser.add_argument("n_trials", type=int, help="number of study trials", default=10)
    args = parser.parse_args()
    name = args.name
    n_trials = args.n_trials
    mp.set_start_method("spawn", force=True)
    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(optuna_run, n_trials=n_trials)
