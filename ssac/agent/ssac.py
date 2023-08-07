import equinox as eqx
import numpy as np
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig

from ssac.agent import safe_actor_critic as sac
from ssac.agent.replay_buffer import ReplayBuffer
from ssac.training.logging import TrainingLogger
from ssac.training.trajectory import Transition
from ssac.types import FloatArray
from ssac.utils import PRNGSequence


@eqx.filter_jit
def policy(actor, observation, key):
    act = lambda o: actor.act(o, key)
    return eqx.filter_vmap(act)(observation)


class SafeSAC:
    def __init__(
        self,
        observation_space: Box | Discrete,
        action_space: Box | Discrete,
        config: DictConfig,
        logger: TrainingLogger,
    ):
        self.prng = PRNGSequence(config.training.seed)
        self.config = config
        self.logger = logger
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(
            config.agent.replay_buffer.capacity,
            config.training.seed,
            config.agent.batch_size,
        )
        self.actor_critic = sac.ActorCritic(
            observation_space, action_space, config, next(self.prng)
        )

    def __call__(self, observation: FloatArray) -> FloatArray:
        if len(self.replay_buffer) > self.config.agent.prefill:
            for batch in self.replay_buffer.sample(self.config.training.parallel_envs):
                losses = self.actor_critic.update(batch, next(self.prng))
                log(losses, self.logger)
            self.actor_critic.polyak(self.config.agent.polyak_rate)
        action = policy(self.actor_critic.actor, observation, next(self.prng))
        return np.asarray(action)

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.store(transition)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = TrainingLogger(self.config.log_dir)


def log(losses, logger):
    logger["critic_loss"] = losses[0].item()
    logger["actor_loss"] = losses[1].item()
    logger["lagrangian_loss"] = losses[2].item()
