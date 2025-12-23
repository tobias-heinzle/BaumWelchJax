import jax
import jax.numpy as jnp

import pytest

from baum_welch_jax.models import HiddenMarkovModel, check_valid_hmm, assert_valid_hmm

def test_hmm_tree_map():
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.eye(2),
        mu = jnp.ones(2) / 2
    )

    result = jax.tree.map(lambda x: x,
        [
        hmm,
        hmm,
        ])
    
    assert jnp.all(result[0].T == jnp.eye(2))
    assert jnp.all(result[0].O == jnp.eye(2))
    assert jnp.all(result[0].mu == jnp.ones(2) / 2)

def test_hmm_assert_T():
    hmm = HiddenMarkovModel(
        T = jnp.ones(2),
        O = jnp.eye(2),
        mu = jnp.ones(2) / 2
    )
    with pytest.raises(ValueError):
        assert_valid_hmm(hmm)

def test_hmm_assert_O():
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.ones(2),
        mu = jnp.ones(2) / 2
    )
    with pytest.raises(ValueError):
        assert_valid_hmm(hmm)

def test_hmm_assert_mu():
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.eye(2),
        mu = jnp.ones(2)
    )
    with pytest.raises(ValueError):
        assert_valid_hmm(hmm)

def test_hmm_jit_validation():
    func = lambda x: check_valid_hmm(HiddenMarkovModel(jnp.eye(2), jnp.eye(2), x))
    func = jax.jit(func)
    func(jnp.zeros(2))

    valid_res = func(jnp.ones(2) / 2)
    invalid_res = func(jnp.ones(2))

    assert valid_res == True
    assert invalid_res == False