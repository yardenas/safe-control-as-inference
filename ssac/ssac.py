from typing import Any
from omegaconf import DictConfig
from gymnasium.spaces import Discrete, Box

from ssac.logging import TrainingLogger
from ssac.trajectory import Transition
from ssac.types import FloatArray


class SafeSAC:
    def __init__(
        self,
        observation_space: Box | Discrete,
        action_space: Box | Discrete,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        self.action_space = action_space

    def __call__(self, observation: FloatArray) -> FloatArray:
        return self.action_space.sample()

    def observe(self, transition: Transition) -> None:
        pass
