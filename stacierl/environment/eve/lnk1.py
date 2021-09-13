from ..envfactory import EnvFactory
import eve
import math


class LNK1(EnvFactory):
    def __init__(self) -> None:
        ...

    def create_env(self) -> eve.Env:
        tree_name = eve.vesseltree.CAD_VesselTrees.ORGANIC_V2
        vessel_tree = eve.vesseltree.CAD(vessel_tree_name=tree_name)
        device = eve.device.Guidewire()
        simulation = eve.simulation.SofaPy3(
            device=device, vessel_tree=vessel_tree, sofa_native_gui=False
        )
        start = eve.start.InsertionPoint()
        target = eve.target.CenterlineRandom(vessel_tree, simulation, target_threshold=10)
        success = eve.success.TargetReached(target)
        pathfinder = eve.pathfinder.Centerline(vessel_tree, simulation, target)

        dim_to_delete = eve.state.wrapper.Dimension.Y
        position = eve.state.Tracking(simulation, vessel_tree, n_points=5)
        position = eve.state.wrapper.RelativeToFirstRow(position)
        position = eve.state.wrapper.CoordinatesTo2D(position, dimension_to_delete=dim_to_delete)
        position = eve.state.wrapper.Normalize(position)
        position_2 = eve.state.Tracking(simulation, vessel_tree, n_points=1)
        position_2 = eve.state.wrapper.CoordinatesTo2D(
            position_2, dimension_to_delete=dim_to_delete
        )
        position_2 = eve.state.wrapper.Normalize(position_2)
        position_2 = eve.state.wrapper.RelativeToLastState(position_2, name="Delta_Tracking")
        target_state = eve.state.Target(target, vessel_tree)
        target_state = eve.state.wrapper.CoordinatesTo2D(
            target_state, dimension_to_delete=dim_to_delete
        )
        target_state = eve.state.wrapper.Normalize(target_state)
        last_action = eve.state.LastAction(simulation)
        state = eve.state.Combination([position, position_2, target_state, last_action])

        target_reward = eve.reward.TargetReached(target, factor=1.0)
        # step_reward = eve.reward.Step(factor=-0.01)
        path_delta = eve.reward.PathLengthDelta(pathfinder, 0.01)
        reward = eve.reward.Combination([target_reward, path_delta])

        max_steps = eve.done.MaxSteps(200)
        target_reached = eve.done.TargetReached(target)
        done = eve.done.Combination([max_steps, target_reached])

        visualisation = eve.visualisation.VisualisationDummy(simulation, vessel_tree)

        return eve.Env(
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
