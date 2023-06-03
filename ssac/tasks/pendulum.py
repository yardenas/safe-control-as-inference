import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import clip_action, rescale_action


def make():
    import gymnasium

    env = gymnasium.make("Pendulum-v1", render_mode="human")
    env = rescale_action.RescaleAction(env, -1.0, 1.0)
    env.action_space = Box(-1.0, 1.0, env.action_space.shape, np.float32)
    env = clip_action.ClipAction(env)
    return env
