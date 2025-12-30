import functools

import jax
from jax.random import key, split
import jax.numpy as jnp

import pytest

from baum_welch_jax.algorithms import baum_welch, generate_sequence
from baum_welch_jax.models import HiddenMarkovParameters, assert_valid_hmm
from baum_welch_jax.util import normalize_rows

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
    hmm = HiddenMarkovParameters(T, O, mu)

    init_guess = HiddenMarkovParameters(
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

@pytest.mark.parametrize('epsilon', [1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_precision(mode, epsilon):
    T = jnp.array([[0.9, 0.1], [0.1,0.9]])
    O = jnp.eye(2)
    mu = jnp.array([0.0, 1.0])
    hmm = HiddenMarkovParameters(T, O, mu)

    init_guess = HiddenMarkovParameters(
        jnp.ones((2,2)) / 2, 
        jnp.ones((2,2)) / 2, 
        jnp.array([0.2,0.8]))

    _, obs = generate_sequence(key(0), hmm, 500)

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=2000, epsilon=epsilon, mode=mode)

    assert result.iterations > 5
    assert result.iterations < 2000
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= 0)

@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_long_sequence(mode):
    T = jnp.array([
        [0.1, 0.0, 0.5, 0.4, 0.0],
        [0.5, 0.1, 0.0, 0.2, 0.2],
        [0.0, 0.9, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.2, 0.0, 0.8],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ])
    O = jnp.array([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.7, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0],
        [0.1, 0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ])
    mu = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])
    hmm = HiddenMarkovParameters(T, O, mu)
    assert_valid_hmm(hmm)

    init_guess = HiddenMarkovParameters(
        normalize_rows(T + 0.25 * jax.random.uniform(key(0), shape=(5,5))), 
        normalize_rows(O + 0.25 * jax.random.uniform(key(1), shape=(5,5))), 
        normalize_rows(jnp.ones(5)))
    assert_valid_hmm(init_guess)
    
    _, obs = generate_sequence(key(9999), hmm, 20_000)

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=2000, epsilon=1e-8, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= 0), f'{result.iterations} iterations performed until monotonicity violated'
    assert jnp.allclose(res_params.T, hmm.T, atol=0.02)
    assert jnp.allclose(res_params.O, hmm.O, atol=0.02)
    assert jnp.allclose(res_params.mu, hmm.mu, atol=0.0001)


@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_mutli_sequence(mode):
    n_seq = 50

    T = jnp.array([[0.9, 0.1], [0.1,0.9]])
    O = jnp.eye(2)
    mu = jnp.array([0.0, 1.0])
    hmm = HiddenMarkovParameters(T, O, mu)
    init_guess = HiddenMarkovParameters(
        jnp.ones((2,2)) / 2, 
        jnp.ones((2,2)) / 2, 
        jnp.array([0.2,0.8])
        )

    _, obs = jax.vmap(lambda _k: generate_sequence(key(_k), hmm, 500))(jnp.arange(n_seq))

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, epsilon=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= 0)
    assert jnp.allclose(res_params.T, hmm.T, atol=0.01)
    assert jnp.allclose(res_params.O, hmm.O, atol=0.005)
    assert jnp.allclose(res_params.mu, hmm.mu, atol=0.000005)

@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_observation_probabilities_structured(mode):

    n_seq = 1000
    seq_keys = split(key(123), n_seq)

    n, m = HMM_TEST_STRUCTURED.O.shape
    
    O_guess = (jnp.ones((n,m)) / m).at[-1].set(jnp.zeros(m).at[-1].set(1.0))
    mu_guess = jnp.zeros(n).at[0].set(1.0)
    init_guess = HiddenMarkovParameters(
        jnp.ones((n,n)) / n, 
        O_guess, 
        mu_guess)

    _, reference_obs = jax.vmap(lambda _k: generate_sequence(_k, HMM_TEST_STRUCTURED, 100))(jnp.array(seq_keys))

    # Run the algorithm
    result = baum_welch(reference_obs[:200], init_guess.astype(jnp.float64), max_iter=1000, epsilon=5e-10, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    # Generate another batch of sequences from the learned parameters 
    _, obs = jax.vmap(lambda _k: generate_sequence(_k, res_params, 100))(jnp.array(seq_keys))
    obs_dist_over_time = jax.lax.map(lambda o: jnp.count_nonzero(obs == o, axis=0), jnp.arange(m)) / n_seq

    assert not result.terminated
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= 0), f'{result.iterations} iterations performed until monotonicity violated'
    assert jnp.allclose(OBS_DISTR_STRUCTURED_100_STEPS, obs_dist_over_time, atol=0.07), f'res_params.O = {res_params.O}'
    

@pytest.mark.slow
@pytest.mark.parametrize('m', [2, 3, 5])
@pytest.mark.parametrize('n', [3, 4, 6])
@pytest.mark.parametrize('seed', [0, 174803])
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_observation_probabilities_random(mode, seed, m, n):
    # Test setup, the goal is to verify if the learned 
    # HMM produces a similar distribution of observations
    n_seq = 100
    key_T, key_O, key_mu, *seq_keys = split(key(seed), 3 + n_seq)

    T = jax.random.uniform(key_T, (n,n))
    O = jax.random.uniform(key_O, (n, m))
    mu = jax.random.uniform(key_mu, n)

    T = T / jnp.sum(T, axis=1)[..., None]
    O = O / jnp.sum(O, axis=1)[..., None]
    mu = mu / jnp.sum(mu)

    hmm = HiddenMarkovParameters(T, O, mu)

    init_guess = HiddenMarkovParameters(
        jnp.ones_like(T) / n, 
        jnp.ones_like(O) / m, 
        jnp.ones_like(mu) / n)

    _, obs = jax.vmap(lambda _k: generate_sequence(_k, hmm, 500))(jnp.array(seq_keys))

    # Run the algorithm
    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, epsilon=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    # Generate another batch of sequences from the learned parameters 
    _, test_obs = jax.vmap(lambda _k: generate_sequence(_k, res_params, 500))(jnp.array(seq_keys))
    mean_obs = jnp.mean(obs)
    mean_test_obs = jnp.mean(test_obs)
    obs_dist = jax.lax.map(lambda o: jnp.mean(obs == o), jnp.arange(m))
    test_obs_dist = jax.lax.map(lambda o: jnp.mean(test_obs == o), jnp.arange(m))

    assert not result.terminated
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= 0)
    assert jnp.allclose(mean_test_obs, mean_obs, rtol=0.01)
    assert jnp.allclose(test_obs_dist, obs_dist, rtol=0.02)