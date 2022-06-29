import stacierl
import eve
import torch
import torch.multiprocessing as mp
from stacierl.replaybuffer.wrapper import filter_database, delete_from_database

def sac_training(
    trainer_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    worker_device=torch.device("cpu"),
    n_worker=6,
    n_trainer=2,
    lr=0.001991743536437494,
    hidden_layers=[255, 255, 255],
    gamma=0.99,
    replay_buffer=1e6,
    training_steps=1e5,
    consecutive_explore_steps=3,
    steps_between_eval=1e4,
    eval_episodes=100,
    batch_size=164,
    heatup=1e4,
    path_reward_factor=0.001,
):

    success = 0.0
    tree_name = eve.vesseltree.CAD_VesselTrees.ORGANIC_V2
    vessel_tree = eve.vesseltree.CAD(vessel_tree_name=tree_name, object_accuracy="high")

    simulation = eve.simulation.Guidewire(
        tip_youngs_modulus=20e3,
        straight_youngs_modulus=80e3,
        beams_between_key_points=(400, 25),
        collision_edges_between_key_points=(100, 20),
        straight_length=485,
        tip_length=15.2,
        sofa_native_gui=False,
        dt_simulation=0.006,
        normalize_action=True,
        velocity_limits=(20, 6.28)
    )

    start = eve.start.InsertionPoint()
    target = eve.target.CenterlineRandom(target_threshold=5)
    success = eve.success.TargetReached()
    pathfinder = eve.pathfinder.BruteForceBFS()

    dim_to_delete = eve.state.wrapper.Dimension.Y
    position = eve.state.Tracking(n_points=5)
    position = eve.state.wrapper.CoordinatesTo2D(position, dimension_to_delete=dim_to_delete)
    position = eve.state.wrapper.Memory(position,
                                        n_steps=3,
                                        reset_mode=eve.state.wrapper.MemoryResetMode(0))
    target_state = eve.state.Target()
    target_state = eve.state.wrapper.CoordinatesTo2D(target_state, dimension_to_delete=dim_to_delete)
    action = eve.state.LastAction()
    action = eve.state.wrapper.Memory(action,
                                      n_steps=2,
                                      reset_mode=eve.state.wrapper.MemoryResetMode(1))
    state = eve.state.Combination([position, target_state, action])

    target_reward = eve.reward.TargetReached(factor=1.0)
    step_reward = eve.reward.Step(factor=-0.0)
    path_delta = eve.reward.PathLengthDelta(path_reward_factor)
    reward = eve.reward.Combination([target_reward, step_reward, path_delta])

    max_steps = eve.done.MaxSteps(200)
    target_reached = eve.done.TargetReached()
    done = eve.done.Combination([max_steps, target_reached])

    visualisation = eve.visualisation.VisualisationDummy()
    env = eve.Env(
        vessel_tree=vessel_tree,
        simulation=simulation,
        start=start,
        target=target,
        success=success,
        state=state,
        reward=reward,
        done=done,
        visualisation=visualisation,
        pathfinder=pathfinder,
    )

    q_net_1 = stacierl.network.QNetwork(hidden_layers)
    q_net_2 = stacierl.network.QNetwork(hidden_layers)
    policy_net = stacierl.network.GaussianPolicy(hidden_layers, env.action_space)
    
    sac_model = stacierl.algo.sacmodel.Vanilla(
        q1=q_net_1,
        q2=q_net_2,
        policy=policy_net,
        learning_rate=lr,
        obs_space=env.observation_space,
        action_space=env.action_space,
    )
    algo = stacierl.algo.SAC(sac_model, action_space=env.action_space, gamma=gamma)
    replay_buffer = stacierl.replaybuffer.VanillaStepShared(replay_buffer, batch_size)

    delete_filter = filter_database(env, success=0.0, episode_length=200)
    delete_from_database(delete_filter)

    load_filter = filter_database(env, success=1.0, episode_length=20)
    replay_buffer = stacierl.replaybuffer.wrapper.LoadFromDB(nb_loaded_episodes=10,
                                                 db_filter=load_filter, 
                                                 wrapped_replaybuffer=replay_buffer, 
                                                 host='10.15.16.238',
                                                 port=65430)
                                                    
    replay_buffer = stacierl.replaybuffer.wrapper.SavetoDB(replay_buffer, env, host="10.15.16.238", port=65430)
    agent = stacierl.agent.Synchron(
        algo, 
        env, 
        env, 
        replay_buffer, 
        consecutive_action_steps=1, 
        trainer_device=trainer_device,
        worker_device=worker_device,
        n_trainer=n_trainer,
        n_worker=n_worker
    )
    
    next_eval_step_limt = steps_between_eval + agent.step_counter.exploration
    training_steps += agent.step_counter.exploration
    agent.heatup(steps=heatup, custom_action_low=[0.0, -1.0])
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

    return success, agent.step_counter.exploration


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    result = sac_training()
