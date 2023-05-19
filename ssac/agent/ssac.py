import numpy as np
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig

from ssac.training.logging import TrainingLogger
from ssac.training.trajectory import Transition
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
        a = np.asarray(self.action_space.sample())
        return np.repeat(a[None], observation.shape[0])

    def observe(self, transition: Transition) -> None:
        pass
