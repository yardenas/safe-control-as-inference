from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy import typing as npt


class Transition(NamedTuple):
    observation: npt.NDArray[Any]
    next_observation: npt.NDArray[Any]
    action: npt.NDArray[Any]
    reward: npt.NDArray[Any]
    terminal: npt.NDArray[Any]
    cost: npt.NDArray[Any]


@dataclass
class Trajectory:
    transitions: list[Transition] = field(default_factory=list)
    frames: list[npt.NDArray[np.float32 | np.int8]] = field(default_factory=list)

    def __len__(self):
        return len(self.transitions)
