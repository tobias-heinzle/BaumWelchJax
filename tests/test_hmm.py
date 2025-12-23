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

def test_hmm_to_log():
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.eye(2),
        mu = jnp.ones(2) / 2
    )

    hmm_log = hmm.to_log()

    assert hmm_log.is_log is True
    assert_valid_hmm(hmm_log)

def test_hmm_to_prob():
    hmm_log = HiddenMarkovModel(
        T = jnp.log(jnp.eye(2)),
        O = jnp.log(jnp.eye(2)),
        mu = jnp.log(jnp.ones(2) / 2),
        is_log=True
    )

    hmm = hmm_log.to_prob()

    assert hmm.is_log is False
    assert_valid_hmm(hmm_log)

def test_hmm_conversion():
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.eye(2),
        mu = jnp.ones(2) / 2
    )
    hmm_log = hmm.to_log()

    with pytest.raises(ValueError):
        hmm_log.to_log()

    with pytest.raises(ValueError):
        hmm.to_prob()

def test_hmm_conversion_jit():
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.eye(2),
        mu = jnp.ones(2) / 2
    )
    hmm_log = hmm.to_log()
    change_mode = lambda x: x.to_log()
    change_mode = jax.jit(change_mode)
    change_mode(hmm)
    converted_hmm = change_mode(hmm)

    with pytest.raises(ValueError):
        change_mode(hmm_log)

    assert converted_hmm.is_log is True
    assert_valid_hmm(converted_hmm)


@pytest.mark.parametrize('is_log', [True, False])
def test_hmm_assert_T(is_log):
    hmm = HiddenMarkovModel(
        T = jnp.ones(2),
        O = jnp.eye(2),
        mu = jnp.ones(2) / 2,
        is_log=is_log
    )
    if is_log:
        hmm = jax.tree.map(lambda x: jnp.log(x), hmm)

    with pytest.raises(ValueError):
        assert_valid_hmm(hmm)

@pytest.mark.parametrize('is_log', [True, False])
def test_hmm_assert_O(is_log):
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.ones(2),
        mu = jnp.ones(2) / 2,
        is_log=is_log
    )
    if is_log:
        hmm = jax.tree.map(lambda x: jnp.log(x), hmm)

    with pytest.raises(ValueError):
        assert_valid_hmm(hmm)

@pytest.mark.parametrize('is_log', [True, False])
def test_hmm_assert_mu(is_log):
    hmm = HiddenMarkovModel(
        T = jnp.eye(2),
        O = jnp.eye(2),
        mu = jnp.ones(2),
        is_log=is_log
    )
    if is_log:
        hmm = jax.tree.map(lambda x: jnp.log(x), hmm)

    with pytest.raises(ValueError):
        assert_valid_hmm(hmm)


@pytest.mark.parametrize('is_log', [True, False])
def test_hmm_jit_validation(is_log):
    T = O = jnp.log(jnp.eye(2)) if is_log else jnp.eye(2)
    func = lambda x: check_valid_hmm(HiddenMarkovModel(T.copy(), O.copy(), x, is_log=is_log))
    func = jax.jit(func)
    func(jnp.zeros(2))
    mu_invalid = jnp.ones(2)
    mu_valid = mu_invalid / 2.0
    if is_log:
        mu_invalid = jnp.log(mu_invalid)
        mu_valid = jnp.log(mu_valid)

    valid_res = func(mu_valid)
    invalid_res = func(mu_invalid)

    assert valid_res == True
    assert invalid_res == False