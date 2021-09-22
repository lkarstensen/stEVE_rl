from ..envfactory import EnvFactory
import tiltmaze
import math


class LNK2(EnvFactory):
    def __init__(self) -> None:
        ...

    def create_env(self) -> tiltmaze.Env:
        maze = tiltmaze.maze.RSS0(episodes_btw_geometry_change=10)
        # maze = tiltmaze.maze.RealisticArch(episodes_btw_geometry_change=10)
        physic2 = tiltmaze.physics.BallGravity(
            maze=maze, ball_radius=10, action_scaling=(200, 200), dt_step=2 / 3
        )
        # maze = tiltmaze.maze.RealisticArch(episodes_btw_geometry_change=10)
        # physic2 = tiltmaze.physics.BallVelocity(maze=maze, action_scaling=(100, 0.5), dt_step= 2/3 )
        target = tiltmaze.target.CenterlineRandom(maze, physic2, 10)
        pathfinder = tiltmaze.pathfinder.NodesBFS(maze, target, physic2)
        start = tiltmaze.start.CenterlineRandom(maze, physic2)
        # start = tiltmaze.start.Origin(physic2)
        imaging = tiltmaze.imaging.ImagingDummy((500, 500), maze, physic2)

        pos = tiltmaze.state.Position(physic2)
        # pos = tiltmaze.state.wrapper.Memory(pos, 2, reset_mode="fill")
        pos = tiltmaze.state.wrapper.Normalize(pos)
        target_state = tiltmaze.state.Target(target)
        target_state = tiltmaze.state.wrapper.Normalize(target_state)
        # action_state = tiltmaze.state.Action(physic2)
        # rotation = tiltmaze.state.Rotation(physic2)
        # state = tiltmaze.state.Combination(
        #    {"position": pos, "rotation": rotation, "target": target_state}
        # )
        state = tiltmaze.state.Combination([pos, target_state])

        target_reward = tiltmaze.reward.TargetReached(1.0, target)
        step_reward = tiltmaze.reward.Step(-0.005)
        path_length_reward = tiltmaze.reward.PathLengthDelta(0.001, pathfinder)
        reward = tiltmaze.reward.Combination([target_reward, step_reward, path_length_reward])

        done_target = tiltmaze.done.TargetReached(target)
        done_steps = tiltmaze.done.MaxSteps(100)
        done = tiltmaze.done.Combination([done_target, done_steps])

        visu = tiltmaze.visualisation.VisualisationDummy(maze, physic2, target)
        info = tiltmaze.info.TargetReached(target=target)
        # visu = tiltmaze.visualisation.PLT(maze, physic2, target)
        success = tiltmaze.success.TargetReached(target)

        return tiltmaze.Env(
            maze=maze,
            state=state,
            reward=reward,
            done=done,
            physic=physic2,
            start=start,
            target=target,
            imaging=imaging,
            pathfinder=pathfinder,
            visualisation=visu,
            info=info,
            success=success,
        )
