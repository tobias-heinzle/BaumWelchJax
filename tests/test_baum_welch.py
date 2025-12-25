from jax.random import key
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import pytest

from baum_welch_jax.algorithms import baum_welch, generate_sequence
from baum_welch_jax.models import HiddenMarkovModel

from conftest import *

@pytest.mark.parametrize('mode', ['log', 'regular'])
def test_trivial_inference(mode):
    T = jnp.array([[0.9, 0.1], [0.1,0.9]])
    O = jnp.eye(2)
    mu = jnp.array([0.5, 0.5])
    hmm = HiddenMarkovModel(T, O, mu)
    if mode == 'log':
        hmm = hmm.to_log()
    init_guess = HiddenMarkovModel(
        jnp.ones((2,2)) / 2, 
        jnp.ones((2,2)) / 2, 
        jnp.array([0.5,0.5]))

    states, obs = generate_sequence(key(0), hmm, 500)
    result = baum_welch(obs, init_guess, max_iter=1000, epsilon=1e-10, mode=mode)

    assert not result.terminated
    assert jnp.all(jnp.diff(result.log_likelihoods) >= 0)
    assert jnp.allclose(result.params.T, hmm.T)
    assert jnp.allclose(result.params.O, hmm.O)
    assert jnp.allclose(result.params.mu, hmm.mu)

    