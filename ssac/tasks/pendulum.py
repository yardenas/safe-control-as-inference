import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import clip_action, rescale_action, transform_reward


def make(cfg):
    import gymnasium

    env = gymnasium.make("Pendulum-v1", render_mode="human")
    env = rescale_action.RescaleAction(env, -1.0, 1.0)
    env.action_space = Box(-1.0, 1.0, env.action_space.shape, np.float32)
    env = clip_action.ClipAction(env)
    env = transform_reward.TransformReward(env, lambda r: cfg.training.scale_reward * r)
    return env
