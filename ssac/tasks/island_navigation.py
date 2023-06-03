from gymnasium import Wrapper
from gymnasium.spaces import Discrete


class IslandNavigationWrapper(Wrapper):
    def __init__(self, env):
        self.env = env
        self.action_space = Discrete(
            self.env.action_space.n,
            self.env.action_space.seed()[0],
            self.env.action_space.min_action.item(),
        )

    def step(self, observation):
        # TODO (yarden): make observation space use one-hot encoding.
        observation, reward, terminated, truncated, info = super().step(observation)
        # Yup...
        cost = -self.env.unwrapped.env.unwrapped._env.environment_data["safety"]
        info["cost"] = cost
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.env.render_mode == "rgb_array":
            array = self.env.render().transpose(2, 1, 0)
            return array
        else:
            return self.env.render()


def make():
    import gym
    import safe_grid_gym  # noqa F401
    from gymnasium.wrappers import compatibility

    env = gym.make("IslandNavigation-v0")
    env = compatibility.EnvCompatibility(env, render_mode="rgb_array")
    env = IslandNavigationWrapper(env)
    return env
