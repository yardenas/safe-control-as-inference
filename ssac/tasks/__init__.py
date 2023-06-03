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
            from ssac.tasks.pendulum import make

            make_env = make
        case _:
            raise NotImplementedError
    return make_env
