import numpy as np
from tqdm import tqdm

from ssac.async_env import AsyncEnv
from ssac.iteration_summary import IterationSummary
from ssac.trajectory import Trajectory, Transition
from ssac import types


def interact(
    agent: types.Agent,
    environment: AsyncEnv,
    num_steps: int,
    render: bool,
) -> IterationSummary:
    observations = environment.reset()
    trajectories = {k: Trajectory() for k in range(environment.num_envs)}
    iteration_summray = IterationSummary()
    step = 0
    with tqdm(
        total=num_steps,
        unit=f"Steps ({environment.num_envs} parallel)",
    ) as pbar:
        while step < num_steps:
            if render:
                add_frames(trajectories, environment.render())
            actions = agent(observations)
            outs = environment.step(actions)
            transition = make_transition(observations, *outs)
            agent.observe(transition)
            add_transition(trajectories, transition)
            *_, terminal, truncated, _ = outs
            done = truncated | terminal
            if done.any():
                assert outs[0] != transition.next_observation
                finalize(trajectories, done, iteration_summray)
            step += environment.num_envs
            pbar.update(environment.num_envs)
            pbar.set_postfix(
                {"reward": transition.rewards.mean(), "cost": transition.costs.mean()}
            )
            observations = outs[0]
    return iteration_summray


def add_frames(trajectories: dict[int, Trajectory], frames: np.ndarray):
    for k, trajectory in trajectories.items():
        trajectory.frames.append(frames[k])


def add_transition(trajectories: dict[int, Trajectory], transition: Transition):
    transposed_transition = list(*zip(*transition))
    for k, trajectory in trajectories.items():
        trajectory.transitions.append(transposed_transition[k])


def finalize(
    trajectories: dict[int, Trajectory],
    dones: np.ndarray,
    iteration_summary: IterationSummary,
):
    for k, done in zip(trajectories.keys(), dones):
        if done:
            iteration_summary.extend(trajectories[k])
            trajectories[k] = Trajectory()


def make_transition(
    observation, next_observations, rewards, truncated, terminal, infos
):
    next_observations = np.array(
        [
            info.get("final_observation", next_observations[i])
            for i, info in enumerate(infos)
        ]
    )
    costs = np.array([info.get("final_info", info).get("cost", 0.0) for info in infos])
    return Transition(
        observation, next_observations, rewards, truncated, terminal, costs
    )
