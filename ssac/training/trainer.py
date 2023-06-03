import os
from typing import List, Optional

import cloudpickle
from omegaconf import DictConfig

from ssac.agent import SafeSAC
from ssac.training import acting, async_env, logging
from ssac.types import Agent, EnvironmentFactory


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: Optional[Agent] = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: Optional[List[int]] = None,
    ):
        self.config = config
        self.agent = agent
        self.make_env = make_env
        self.epoch = start_epoch
        self.step = step
        self.seeds = seeds
        self.logger: Optional[logging.TrainingLogger] = None
        self.state_writer: Optional[logging.StateWriter] = None
        self.env: Optional[async_env.AsyncEnv] = None

    def __enter__(self):
        log_path = os.getcwd()
        self.logger = logging.TrainingLogger(log_path)
        self.state_writer = logging.StateWriter(log_path)
        self.env = async_env.AsyncEnv(
            self.make_env,
            self.config.training.time_limit,
            self.config.training.parallel_envs,
        )
        if self.seeds is not None:
            self.env.reset(seed=self.seeds)
        else:
            self.env.reset(seed=self.config.training.seed)
        if self.agent is None:
            self.agent = SafeSAC(
                self.env.observation_space,
                self.env.action_space,
                self.config,
                self.logger,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.close()
        self.logger.close()

    def train(self, epochs: Optional[int] = None) -> None:
        epoch, logger, state_writer = self.epoch, self.logger, self.state_writer
        assert logger is not None and state_writer is not None
        for epoch in range(epoch, epochs or self.config.training.epochs):
            print(f"Training epoch #{epoch}")
            self._step(
                train=True,
                num_episodes_per_epoch=self.config.training.episodes_per_epoch,
                prefix="train",
            )
            if (epoch + 1) % self.config.training.eval_every == 0:
                print("Evaluating...")
                self._step(
                    train=False,
                    num_episodes_per_epoch=self.config.training.evaluation_episodes,
                    prefix="evaluate",
                )
            self.epoch = epoch + 1
            state_writer.write(self.state)
        logger.flush()

    def _step(
        self,
        train: bool,
        num_episodes_per_epoch: int,
        prefix: str,
    ) -> None:
        config, agent, env, logger = self.config, self.agent, self.env, self.logger
        assert env is not None and agent is not None and logger is not None
        render_episodes = int(not train) * self.config.training.render_episodes
        summary = acting.interact(
            agent,
            env,
            num_episodes_per_epoch,
            render_episodes,
        )
        if train:
            self.step += num_episodes_per_epoch
        objective, cost_rate, feasibilty = summary.metrics
        logger.log_summary(
            {
                f"{prefix}/objective": objective,
                f"{prefix}/cost_rate": cost_rate,
                f"{prefix}/feasibility": feasibilty,
            },
            self.step,
        )
        if render_episodes > 0:
            num_log = min(5, config.training.parallel_envs)
            logger.log_video(
                summary.videos[:num_log],
                self.step,
                "video",
                30,
            )
        logger.log_metrics(self.step)

    def get_env_random_state(self):
        assert self.env is not None
        rs = [
            state.get_state()[1]
            for state in self.env.get_attr("rs")
            if state is not None
        ]
        if not rs:
            rs = [
                state.bit_generator.state["state"]["state"]
                for state in self.env.get_attr("np_random")
            ]
        return rs

    @classmethod
    def from_pickle(cls, config: DictConfig) -> "Trainer":
        log_path = config.log_dir
        with open(os.path.join(log_path, "state.pkl"), "rb") as f:
            (
                make_env,
                env_rs,
                agent,
                epoch,
                step,
            ) = cloudpickle.load(f).values()
        print(f"Resuming experiment from: {log_path}...")
        assert agent.config == config, "Loaded different hyperparameters."
        return cls(
            config=agent.config,
            make_env=make_env,
            start_epoch=epoch,
            seeds=env_rs,
            agent=agent,
        )

    @property
    def state(self):
        return {
            "make_env": self.make_env,
            "env_rs": self.get_env_random_state(),
            "agent": self.agent,
            "epoch": self.epoch,
            "step": self.step,
        }
