from typing import NamedTuple, Optional

import distrax as trx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig
from optax import OptState, l2_loss

from ssac.training.trajectory import Transition
from ssac.utils import Learner


def inv_softplus(x):
    return jnp.where(x < 20.0, jnp.log(jnp.exp(x) - 1.0), x)


class ActorBase(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int,
        hidden_size: int,
        *,
        key: jax.random.KeyArray
    ):
        self.net = eqx.nn.MLP(state_dim, action_dim, hidden_size, n_layers, key=key)

    def __call__(self, state: jax.Array) -> trx.Distribution:
        raise NotImplementedError

    def act(
        self,
        state: jax.Array,
        key: Optional[jax.random.KeyArray] = None,
        deterministic: bool = False,
    ) -> jax.Array:
        if deterministic:
            return self(state).mean()
        else:
            assert key is not None
            return self(state).sample(seed=key)


class ContinuousActor(ActorBase):
    net: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int,
        hidden_size: int,
        *,
        key: jax.random.KeyArray
    ):
        self.net = eqx.nn.MLP(state_dim, action_dim * 2, hidden_size, n_layers, key=key)

    def __call__(self, state: jax.Array) -> trx.Normal:
        x = self.net(state)
        mu, stddev = jnp.split(x, 2, axis=-1)
        init_std = inv_softplus(5.0)
        stddev = jnn.softplus(stddev + init_std) + 0.1
        dist = trx.Normal(mu, stddev)
        dist = trx.Transformed(dist, trx.Tanh())
        return dist


class DiscreteActor(ActorBase):
    net: eqx.nn.MLP

    def __call__(self, state: jax.Array) -> trx.Categorical:
        logits = self.net(state)
        # TODO (yarden): not sure if distrax's implementation uses
        # reparameterization/relaxation for taking gradients
        #  through samples of Categorical
        return trx.Categorical(logits=logits)


class Critic(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        key: jax.random.KeyArray
    ):
        self.net = eqx.nn.MLP(
            state_dim + (action_dim if action_dim is not None else 0),
            1,
            hidden_size,
            n_layers,
            key=key,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        x = self.net(jnp.concatenate([state, action], axis=-1))
        return x.squeeze(-1)


class CriticPair(NamedTuple):
    uno_critic: Critic
    dos_critic: Critic

    def __call__(
        self, state: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return self.uno_critic(state, action), self.dos_critic(state, action)


def make_actor(
    observation_space: Box | Discrete,
    action_space: Box | Discrete,
    config: DictConfig,
    key: jax.random.KeyArray,
) -> ContinuousActor | DiscreteActor:
    state_dim = np.prod(observation_space.shape)  # type: ignore
    if isinstance(action_space, Box):
        action_dim = np.prod(action_space.shape)  # type: ignore
        actor = ContinuousActor(
            state_dim=state_dim, action_dim=action_dim, **config.actor, key=key
        )
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
        actor = DiscreteActor(
            state_dim=state_dim, action_dim=action_dim, **config.actor, key=key
        )
    else:
        raise NotImplementedError
    return actor


def make_critics(
    observation_space: Box | Discrete,
    action_space: Box | Discrete,
    config: DictConfig,
    key: jax.random.KeyArray,
):
    state_dim = np.prod(observation_space.shape)  # type: ignore
    if isinstance(action_space, Box):
        action_dim = np.prod(action_space.shape)  # type: ignore
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
    else:
        raise NotImplementedError
    critic_factory = lambda key: Critic(
        state_dim=state_dim, action_dim=action_dim, **config.actor, key=key
    )
    key1, key2 = jax.random.split(key)
    one = critic_factory(key1)
    two = critic_factory(key2)
    return CriticPair(one, two)


class ActorCritic:
    def __init__(
        self,
        observation_space: Box | Discrete,
        action_space: Box | Discrete,
        config: DictConfig,
        key: jax.random.KeyArray,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = make_actor(observation_space, action_space, config, actor_key)
        self.actor_learner = Learner(self.actor, config.actor_optimizer)
        self.critics = make_critics(observation_space, action_space, config, critic_key)
        self.critics_learner = Learner(self.critics, config.critic_optimizer)
        self.discount = config.agent.discount
        self.log_lagrangians = jnp.asarray(config.agent.initial_log_lagrangians)
        self.log_lagrangians_learner = Learner(
            self.log_lagrangians, config.lagrangians_optimizer
        )
        self.target_entropy = np.prod(action_space.shape)  # type: ignore
        self.target_safety = config.training.cost_limit

    def update(self, batch: Transition, key: jax.random.KeyArray):
        actor_key, critics_key = jax.random.split(key)
        lagrangians = jnp.exp(self.log_lagrangians)
        self.critics, self.critics_learner.state, critic_loss = update_critics(
            batch,
            self.critics,
            self.actor,
            self.critics_learner.state,
            self.critics_learner,
            lagrangians,
            self.discount,
            critics_key,
        )
        self.actor, self.actor_learner.state, rest = update_actor(
            batch,
            self.actor,
            self.critics,
            self.actor_learner.state,
            self.actor_learner,
            self,
            lagrangians,
            actor_key,
        )
        (
            self.log_lagrangians,
            self.log_lagrangians_learner.state,
            lagrangian_loss,
        ) = update_log_lagrangians(
            self.log_lagrangians,
            -rest["log_pi"],
            jnp.asarray(self.target_entropy),
        )


@eqx.filter_jit
def update_critics(
    batch: Transition,
    critics: CriticPair,
    actor: ContinuousActor | DiscreteActor,
    learning_state: OptState,
    learner: Learner,
    lagrangians: jax.Array,
    discount: float,
    key: jax.random.KeyArray,
):
    def loss_fn(critics):
        next_action, log_pi = actor(batch.next_observation).sample_and_log_prob(
            seed=key
        )
        next_qs = critics(batch.next_observation, next_action)
        debiased_q = jnp.minimum(*next_qs)
        bonus = -lagrangians.dot(log_pi)
        next_soft_q = debiased_q + bonus
        soft_q_target = batch.reward + (1.0 - batch.terminal) * discount * next_soft_q
        soft_q_target = jax.lax.stop_gradient(soft_q_target)
        q1, q2 = critics(batch.observation, batch.action)
        return (l2_loss(q1, soft_q_target) + l2_loss(q2, soft_q_target)).mean()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(critics)
    new_critics, new_learning_state = learner.grad_step(critics, grads, learning_state)
    return (new_critics, new_learning_state), loss


@eqx.filter_jit
def update_actor(
    batch: Transition,
    actor: ContinuousActor | DiscreteActor,
    critics: CriticPair,
    learning_state: OptState,
    learner: Learner,
    lagrangians: jax.Array,
    key: jax.random.KeyArray,
):
    def loss_fn(actor):
        action, log_pi = actor(batch.observation).sample_and_log_prob(seed=key)
        qs = critics(batch.observation, action)
        debiased_q = jnp.minimum(*qs)
        bonus = -lagrangians.dot(log_pi)
        objective = debiased_q + bonus
        return -objective.mean(), log_pi.mean()

    (loss, log_pi), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(actor)
    new_actor, new_learning_state = learner.grad_step(actor, grads, learning_state)
    return (new_actor, new_learning_state), dict(loss=loss, log_pi=log_pi)


@eqx.filter_jit
def update_log_lagrangians(
    log_lagrangians: jax.Array,
    constraints: jax.Array,
    targets: jax.Array,
    learning_state: OptState,
    learner: Learner,
):
    def loss_fn(log_lagrangians):
        lagrangians = jnp.exp(log_lagrangians)
        penalty = lagrangians * (constraints - targets)
        return penalty.sum()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(log_lagrangians)
    new_log_lagrangians, new_learning_state = learner.grad_step(
        log_lagrangians, grads, learning_state
    )
    return (new_log_lagrangians, new_learning_state), loss