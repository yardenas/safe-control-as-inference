from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy import typing as npt

from ssac.trajectory import Trajectory


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
        for trajectory in self._data:
            if len(trajectory.frames) > 0:
                all_vids.append(trajectory.frames)
        vids = np.asarray(all_vids)[-1].swapaxes(0, 1)
        return vids

    def extend(self, samples: Trajectory) -> None:
        self._data.append(samples)


def _objective(rewards: npt.NDArray[Any]) -> float:
    return float(rewards.sum(2).mean())


def _cost_rate(costs: npt.NDArray[Any]) -> float:
    return float(costs.mean())


def _feasibility(costs: npt.NDArray[Any], boundary: float) -> float:
    return float((costs.sum(2).mean(1) <= boundary).mean())
