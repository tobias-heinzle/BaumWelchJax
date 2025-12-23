from jax.random import key
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import pytest

from baum_welch_jax.algorithms import baum_welch, generate_sequence
from baum_welch_jax.models import HiddenMarkovModel

from conftest import *

@pytest.mark.parametrize('mode', ['log', 'regular'])
def test_basic_inference(mode):
    T = jnp.zeros((2,2))
    T = T.at[:,1].set(1.0)
    O = jnp.eye(2)
    mu = jnp.zeros(2)
    mu = mu.at[0].set(1.0)
    hmm = HiddenMarkovModel(T, O, mu)
    if mode == 'log':
        hmm = hmm.to_log()

    states, obs = generate_sequence(key(0), hmm, 5)

    result = baum_welch(obs, hmm, mode=mode)

    assert jnp.allclose(result.params.T, hmm.T)
    assert jnp.allclose(result.params.O, hmm.O)
    assert jnp.allclose(result.params.mu, hmm.mu)

    