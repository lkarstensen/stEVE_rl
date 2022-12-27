import stacierl
import eve
import torch
from datetime import datetime
import csv
import os
import torch.optim as optim


def sac_training(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr=0.001991743536437494,
    hidden_layers=[255, 255, 255],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=1e5,
    consecutive_explore_steps=1,
    steps_between_eval=1e4,
    eval_episodes=100,
    batch_size=164,
    heatup=1e4,
    log_folder: str = os.getcwd() + "/fast_learner_example_results/",
    id=0,
    name="fast_learner",
    *args,
    **kwargs,
):

    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    success = 0.0
    vessel_tree = eve.vesseltree.AorticArch(
        seed=1234, rotate_z=20, rotate_x=-5, omit_y_axis=True
    )
    vessel_tree.scale(1, 1, 1, 0.5)

    instrument = eve.simulation2d.device.JWire()

    simu = eve.simulation2d.SingleDevice(
        vessel_tree,
        instrument,
        velocity_limit=(50, 1.5),
        element_length=1.75,
        image_frequency=7.5,
        dt_simulation=0.0002,
        friction=1.0,
        damping=0.000001,
        body_mass=0.01,
        body_moment=0.1,
        linear_stiffness=2.5e6,
        linear_damping=100,
        last_segment_kp_angle=2,
        last_segment_kp_translation=5,
    )

    target = eve.target.CenterlineRandom(
        vessel_tree,
        simu,
        5,
        branches=[
            "right subclavian artery",
            "right common carotid artery",
            "left common carotid artery",
            "left subclavian artery",
            "brachiocephalic trunk",
        ],
    )
    pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simu, target)
    start = eve.start.InsertionPoint(simu)
    imaging = eve.imaging.ImagingDummy((500, 500))

    pos = eve.state.Tracking(simu, n_points=6, resolution=1)
    pos = eve.state.wrapper.Normalize(pos)
    target_state = eve.state.Target(target)
    target_state = eve.state.wrapper.Normalize(target_state)
    rot = eve.state.Rotations(simu)
    state = eve.state.Combination([pos, target_state, rot])

    target_reward = eve.reward.TargetReached(target, 1.0)
    step_reward = eve.reward.Step(-0.005)
    path_length_reward = eve.reward.PathLengthDelta(pathfinder, 0.001)
    reward = eve.reward.Combination([target_reward, step_reward, path_length_reward])

    done_target = eve.done.TargetReached(target)
    done_steps = eve.done.MaxSteps(100)
    done = eve.done.Combination([done_target, done_steps])

    success = eve.success.TargetReached(target)
    visu = eve.visualisation.VisualisationDummy()

    env = eve.Env(
        vessel_tree=vessel_tree,
        state=state,
        reward=reward,
        done=done,
        intervention=simu,
        start=start,
        target=target,
        imaging=imaging,
        pathfinder=pathfinder,
        visualisation=visu,
        success=success,
    )

    n_observations = env.observation_space.n_observations
    n_actions = env.action_space.n_actions
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
        n_observations=env.observation_space.n_observations,
        n_actions=env.action_space.n_actions,
    )
    algo = stacierl.algo.SAC(
        sac_model, n_actions=env.action_space.n_actions, gamma=gamma
    )
    replay_buffer = stacierl.replaybuffer.VanillaStep(replay_buffer, batch_size)
    agent = stacierl.agent.Single(
        algo,
        env,
        env,
        replay_buffer,
        consecutive_action_steps=1,
        device=device,
        normalize_actions=True,
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
            # agent.save_checkpoint(log_folder, f"checkpoint_{step_counter.exploration}")
            episodes = agent.evaluate(episodes=eval_episodes)
            rewards = [episode.episode_reward for episode in episodes]
            successes = [episode.episode_success for episode in episodes]
            reward = sum(rewards) / len(rewards)
            success = sum(successes) / len(successes)
            next_eval_step_limt += steps_between_eval

            print(
                f"Steps: {step_counter.exploration}, Reward: {reward}, Success: {success}"
            )
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

    return success, agent.step_counter.exploration


if __name__ == "__main__":
    result = sac_training()
