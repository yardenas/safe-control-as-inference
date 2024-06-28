from gymnasium.wrappers.compatibility import EnvCompatibility
from omegaconf import DictConfig
import numpy as np
#from ssac.benchmark_suites.utils import get_domain_and_task
def get_domain_and_task(cfg: DictConfig) -> tuple[str, DictConfig]:
    assert len(cfg.environment.keys()) == 1
    domain_name, task = list(cfg.environment.items())[0]
    return domain_name, task

from ssac.rl.types import EnvironmentFactory
from ssac.rl.wrappers import ChannelFirst
#robot_name = "point" and task = "go_to_goal"

def sample_task(seed):
    easy_tasks = (
        "go_to_goal",
        "push_box",
        "collect",
        "catch_goal",
        "press_buttons",
        "unsupervised",
    )
    task = np.random.RandomState(seed).choice(easy_tasks)
    return task


class SafeAdaptationEnvCompatibility(EnvCompatibility):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        return self.env.reset(options=options), {}

#safe_gym_point_to_goal
def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env(cfg):
        import safe_adaptation_gym

        '''_, task_cfg = get_domain_and_task(cfg)
        if task_cfg.task is not None:
            task = task_cfg.task
        else:
            task = sample_task(cfg.training.seed)'''
        #hardcode environment specification
        robot_name = "point"
        task = "go_to_goal"
        env = safe_adaptation_gym.make(
            robot_name=robot_name,
            task_name=task,
            seed=cfg.training.seed,
            rgb_observation=False, #does this change the shape of the observation??
            render_lidar_and_collision=False,
        )
        env = SafeAdaptationEnvCompatibility(env)
        '''if (
            task_cfg.image_observation.enabled
            and task_cfg.image_observation.image_format == "channels_first"
        ):
            env = ChannelFirst(env)'''
        return env

    return make_env  # type: ignore