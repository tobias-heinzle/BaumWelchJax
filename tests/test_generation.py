from jax.random import key
import jax.numpy as jnp

import pytest

from baum_welch_jax.algorithms import generate_sequence
from baum_welch_jax.models import HiddenMarkovParameters, assert_valid_hmm


@pytest.mark.parametrize('start_state', [0,1])
def test_generation_sanity_check(start_state):
    mu = jnp.zeros(2)
    mu = mu.at[start_state].set(1.0)
    simple_hmm = HiddenMarkovParameters(jnp.eye(2), jnp.eye(2), mu)
    states, observations = generate_sequence(key(0), simple_hmm, 10)

    assert jnp.allclose(states, start_state)
    assert jnp.allclose(observations, start_state)


@pytest.mark.parametrize('start_state', [0,1])
def test_generation_rotated_observations(start_state):
    mu = jnp.zeros(2)
    mu = mu.at[start_state].set(1.0)
    simple_hmm = HiddenMarkovParameters(jnp.eye(2), jnp.rot90(jnp.eye(2)), mu)
    states, observations = generate_sequence(key(0), simple_hmm, 10)

    assert jnp.allclose(states, start_state)
    assert jnp.allclose(observations, 1 - start_state)


def test_generation_sequential_sanity_check():
    T = jnp.eye(10, k=1)
    T = T.at[-1,-1].set(1.0)
    O = jnp.eye(10)
    mu = jnp.array([1.0] + [0.0]*9)
    simple_hmm = HiddenMarkovParameters(T, O, mu)
    assert_valid_hmm(simple_hmm)
    states, observations = generate_sequence(key(0), simple_hmm, 10)

    assert jnp.allclose(states, jnp.arange(10))
    assert jnp.allclose(observations, jnp.arange(10))


def test_generation_sequential_rotated_observations():
    T = jnp.eye(10, k=1)
    T = T.at[-1,-1].set(1.0)
    O = jnp.rot90(jnp.eye(10))
    mu = jnp.array([1.0] + [0.0]*9)
    simple_hmm = HiddenMarkovParameters(T, O, mu)
    assert_valid_hmm(simple_hmm)
    states, observations = generate_sequence(key(0), simple_hmm, 10)

    assert jnp.allclose(states, jnp.arange(10))
    assert jnp.allclose(observations, jnp.arange(10)[::-1])


def test_generation_parallel_sanity_check():
    T = jnp.eye(20)
    O = jnp.eye(20)
    mu = jnp.eye(20)
    simple_hmm = HiddenMarkovParameters(T, O, mu)
    assert_valid_hmm(simple_hmm)
    states, observations = generate_sequence(key(0), simple_hmm, 10)

    test_outcome = jnp.array([jnp.full(10, k) for k in range(20)])
    assert jnp.allclose(states, test_outcome)
    assert jnp.allclose(observations, test_outcome)


def test_generation_parallel_rotated_observations():
    T = jnp.eye(20)
    O = jnp.rot90(jnp.eye(20)) 
    mu = jnp.eye(20)
    simple_hmm = HiddenMarkovParameters(T, O, mu)
    assert_valid_hmm(simple_hmm)
    states, observations = generate_sequence(key(0), simple_hmm, 10)

    test_outcome = jnp.array([jnp.full(10, k) for k in range(20)])
    assert jnp.allclose(states, test_outcome)
    assert jnp.allclose(observations, test_outcome[::-1])