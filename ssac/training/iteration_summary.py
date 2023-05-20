from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import typing as npt

from ssac.training.trajectory import Trajectory


@dataclass
class IterationSummary:
    _data: list[Trajectory] = field(default_factory=list)
    cost_boundary: float = 25.0

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def metrics(self) -> tuple[float, float, float]:
        rewards, costs = [], []
        for trajectory in self._data:
            r = sum(t.reward for t in trajectory.transitions)
            c = sum(t.cost for t in trajectory.transitions)
            rewards.append(r)
            costs.append(c)
        # Stack data from all tasks on the first axis,
        # giving a [#tasks, #episodes, #time, ...] shape.
        stacked_rewards = np.stack(rewards)
        stacked_costs = np.stack(costs)
        return (
            _objective(stacked_rewards),
            _cost_rate(stacked_costs),
            _feasibility(stacked_costs, self.cost_boundary),
        )

    @property
    def videos(self):
        all_vids = []
        max_len = 0
        for trajectory in self._data:
            if len(trajectory.frames) > 0:
                max_len = max(max_len, len(trajectory.frames))
                all_vids.append(trajectory.frames)
        for i, vid in enumerate(all_vids):
            current_len = len(vid)
            if current_len < max_len:
                all_vids[i] = vid + [vid[-1]] * (max_len - current_len)
        vids = np.asarray(all_vids)
        return vids

    def extend(self, samples: Trajectory) -> None:
        self._data.append(samples)


def _objective(rewards: npt.NDArray[Any]) -> float:
    return float(rewards.mean())


def _cost_rate(costs: npt.NDArray[Any]) -> float:
    return float(costs.mean())


def _feasibility(costs: npt.NDArray[Any], boundary: float) -> float:
    return float((costs.mean() <= boundary).mean())
