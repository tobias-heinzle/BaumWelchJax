import functools

from jax.random import key
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import pytest

from baum_welch_jax.algorithms import baum_welch, generate_sequence
from baum_welch_jax.models import HiddenMarkovModel

from conftest import *

def enable_x64(test_fn):
    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        with jax.enable_x64():
            return test_fn(*args, **kwargs)
    return wrapper

@pytest.mark.parametrize('mode', ['regular', 'log'])
def test_forced_x64(mode):
    with pytest.raises(RuntimeError):
        baum_welch(jnp.zeros(10, dtype=jnp.int32), HMM_TRIVIAL, mode=mode)

@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_trivial_inference(mode):
    T = jnp.array([[0.9, 0.1], [0.1,0.9]])
    O = jnp.eye(2)
    mu = jnp.array([0.0, 1.0])
    hmm = HiddenMarkovModel(T, O, mu)

    init_guess = HiddenMarkovModel(
        jnp.ones((2,2)) / 2, 
        jnp.ones((2,2)) / 2, 
        jnp.array([0.2,0.8]))

    _, obs = generate_sequence(key(0), hmm, 500)

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, epsilon=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= 0)
    assert jnp.allclose(res_params.T, hmm.T, atol=0.05)
    assert jnp.allclose(res_params.O, hmm.O, atol=0.001)
    assert jnp.allclose(res_params.mu, hmm.mu, atol=0.00001)

    