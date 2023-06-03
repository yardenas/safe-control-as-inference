from omegaconf import DictConfig

from ssac.types import EnvironmentFactory


def make(cfg: DictConfig) -> EnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    env = list(cfg.environment.keys())[0]
    match env:
        case "safety_grid":
            from ssac.tasks.island_navigation import make

            make_env = make
        case "pendulum":
            from functools import partial

            import gymnasium

            return partial(gymnasium.make, "Pendulum-v1", render_mode="rgb_array")
        case _:
            raise NotImplementedError
    return make_env
