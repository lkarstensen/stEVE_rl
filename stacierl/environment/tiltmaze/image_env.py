from tiltmaze.state import image
from ..envfactory import EnvFactory
import tiltmaze
import math


class LNK2(EnvFactory):
    def __init__(self, dt_step=1 / 10) -> None:
        self.dt_step = dt_step

    def create_env(self) -> tiltmaze.Env:
        maze = tiltmaze.maze.RSS0(episodes_btw_geometry_change=math.inf, seed=1234)
        velocity_limits = (100, 0.5)
        physic = tiltmaze.physics.BallVelocity(
            maze=maze,
            velocity_limits=velocity_limits,
            action_scaling=velocity_limits,
            dt_step=self.dt_step,
        )
        target = tiltmaze.target.CenterlineRandom(maze, physic, 10)
        pathfinder = tiltmaze.pathfinder.NodesBFS(maze, target, physic)
        start = tiltmaze.start.Origin(physic)
        imaging = tiltmaze.imaging.ImagingDummy()
        imaging = tiltmaze.imaging.LNK1((500, 500), maze, physic)

        pos = tiltmaze.state.Position(physic)
        pos = tiltmaze.state.wrapper.Normalize(pos)
        image_state = tiltmaze.state.Image(imaging)
        image_state = tiltmaze.state.imagewrapper.ShowCorridors(image_state,maze)
        target_state = tiltmaze.state.Target(target)
        target_state = tiltmaze.state.wrapper.Normalize(target_state)
        rotation = tiltmaze.state.Rotation(physic)
        state = tiltmaze.state.Combination(
            {"image":image_state}
        )
        
        #state = tiltmaze.state.Image(imaging)
        #state = tiltmaze.state.imagewrapper.ShowCorridors(state,maze)

        target_reward = tiltmaze.reward.TargetReached(1.0, target)
        step_reward = tiltmaze.reward.Step(-0.005)
        path_length_reward = tiltmaze.reward.PathLengthDelta(0.001, pathfinder)
        reward = tiltmaze.reward.Combination([target_reward, step_reward, path_length_reward])
        done_target = tiltmaze.done.TargetReached(target)
        done_steps = tiltmaze.done.MaxSteps(100)
        done = tiltmaze.done.Combination([done_target, done_steps])

        success = tiltmaze.success.TargetReached(target)
        visu = tiltmaze.visualisation.VisualisationDummy(maze, physic, target)
        visu = tiltmaze.visualisation.PLT(maze,physic,target)
        
        return tiltmaze.Env(
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
