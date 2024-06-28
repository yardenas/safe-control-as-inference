from omegaconf import DictConfig 

from ssac.rl.types import EnvironmentFactory


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
        case "dm_cartpole_swingup":
            from ssac.tasks.dm_cartpole_swingup import make

            make_env = make(cfg) #P: dm_cartpole make fct is weird nested function!
        case "dm_cartpole_balance":
            from ssac.tasks.dm_cartpole_balance import make

            make_env = make(cfg) #P: dm_cartpole make fct is weird nested function!
        case "safe_gym_point_to_goal":
            from ssac.tasks.safe_gym_point_to_goal import make

            make_env = make(cfg) #P: dm_cartpole make fct is weird nested function!
        case _:
            raise NotImplementedError
    return lambda: make_env(cfg)
