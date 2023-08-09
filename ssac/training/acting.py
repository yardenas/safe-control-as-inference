import numpy as np
from tqdm import tqdm

from ssac import types
from ssac.training.async_env import AsyncEnv
from ssac.training.iteration_summary import IterationSummary
from ssac.training.trajectory import Trajectory, Transition


def interact(
    agent: types.Agent,
    environment: AsyncEnv,
    num_episodes: int,
    render: bool,
) -> IterationSummary:
    observations = environment.reset()
    trajectories = {k: Trajectory() for k in range(environment.num_envs)}
    iteration_summray = IterationSummary()
    episodes = 0
    with tqdm(
        total=num_episodes,
        unit=f"Episodes ({environment.num_envs} parallel)",
    ) as pbar:
        while episodes < num_episodes:
            if render:
                add_frames(trajectories, environment.render())
            actions = agent(observations)
            next_obs, reward, terminal, truncated, info = environment.step(actions)
            transition = make_transition(
                observations, next_obs, actions, reward, terminal, info
            )
            agent.observe(transition)
            add_transition(trajectories, transition)
            done = truncated | terminal
            if done.any():
                r, c = finalize(trajectories, done, iteration_summray)
                count = done.sum()
                episodes += count
                pbar.update(count)
                pbar.set_postfix({"reward": r, "cost": c})
                agent.logger.log_metrics(len(agent.replay_buffer))
            observations = next_obs
    assert not iteration_summray.empty
    return iteration_summray


def add_frames(trajectories: dict[int, Trajectory], frames: np.ndarray):
    for k, trajectory in trajectories.items():
        if k > frames.shape[0] - 1:
            break
        trajectory.frames.append(frames[k])


def add_transition(trajectories: dict[int, Trajectory], transition: Transition):
    transposed_transition = list(Transition(*t) for t in zip(*transition))
    for k, trajectory in trajectories.items():
        trajectory.transitions.append(transposed_transition[k])


def finalize(
    trajectories: dict[int, Trajectory],
    dones: np.ndarray,
    iteration_summary: IterationSummary,
) -> tuple[float, float]:
    rs = []
    cs = []
    for k, done in zip(trajectories.keys(), dones):
        if done:
            rs.append(sum(t.reward for t in trajectories[k].transitions))
            cs.append(sum(t.cost for t in trajectories[k].transitions))
            iteration_summary.extend(trajectories[k])
            trajectories[k] = Trajectory()
    return np.asarray(rs).mean(), np.asarray(cs).mean()


def make_transition(observation, next_observations, action, rewards, terminal, infos):
    next_obs = np.array(
        [
            info.get("final_observation", next_observations[i])
            for i, info in enumerate(infos)
        ]
    )
    costs = np.array([info.get("final_info", info).get("cost", 0.0) for info in infos])
    return Transition(observation, next_obs, action, rewards, terminal, costs)
