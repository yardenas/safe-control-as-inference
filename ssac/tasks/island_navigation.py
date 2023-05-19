from gymnasium import Wrapper


class IslandNavigationWrapper(Wrapper):
    def __init__(self, env):
        self.env = env

    def step(self, observation):
        observation, reward, terminated, truncated, info = super().step(observation)
        # Yup...
        cost = -self.env.unwrapped.env.unwrapped._env.environment_data.get("safety")
        info["cost"] = cost
        return observation, reward, terminated, truncated, info


def make():
    import gym
    import safe_grid_gym  # noqa F401
    from gymnasium.wrappers import compatibility

    env = gym.make("IslandNavigation-v0")
    env = compatibility.EnvCompatibility(env, render_mode="rgb_array")
    env = IslandNavigationWrapper(env)
    return env
