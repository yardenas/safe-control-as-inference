from typing import TYPE_CHECKING, Callable, Protocol, Union

from numpy import typing as npt
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

FloatArray = npt.NDArray[Union[np.float32, np.float64]]

if TYPE_CHECKING:
    from ssac.trajectory import Transition


class Agent(Protocol):
    def __call__(self, observation: FloatArray) -> FloatArray:
        ...

    def observe(self, transition: "Transition") -> None:
        ...


EnvironmentFactory = Callable[[], Union[Env[Box, Box], Env[Box, Discrete]]]
