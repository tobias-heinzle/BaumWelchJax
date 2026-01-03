import functools

import jax
from jax.random import key, split, uniform
import jax.numpy as jnp

import pytest

from baum_welch_jax import PrecisionWarning
from baum_welch_jax.algorithms import baum_welch, generate_sequence
from baum_welch_jax.models import HiddenMarkovParameters, assert_valid_hmm, FreezeConfig
from baum_welch_jax.util import normalize_rows

from conftest import *

MONOTONICITY_TOLERANCE = -1e-8

def enable_x64(test_fn):
    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        with jax.enable_x64():
            return test_fn(*args, **kwargs)
    return wrapper

@pytest.mark.parametrize('mode', ['regular', 'log'])
def test_forced_x64(mode):
    with pytest.warns(PrecisionWarning):
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

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
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

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=2000, tol=epsilon, mode=mode)

    assert not result.terminated
    assert result.iterations > 5
    assert result.iterations < 2000
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)

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

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=125, tol=1e-3, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE), f'{result.iterations} iterations'
    assert jnp.allclose(res_params.T, hmm.T, atol=0.02), f'{result.iterations} iterations, T max error: {jnp.max(jnp.abs(res_params.T - hmm.T))}'
    assert jnp.allclose(res_params.O, hmm.O, atol=0.02), f'{result.iterations} iterations, O max error: {jnp.max(jnp.abs(res_params.O - hmm.O))}'
    assert jnp.allclose(res_params.mu, hmm.mu, atol=0.0001), f'{result.iterations} iterations, mu max error: {jnp.max(jnp.abs(res_params.mu - hmm.mu))}'


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

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
    assert jnp.allclose(res_params.T, hmm.T, atol=0.01)
    assert jnp.allclose(res_params.O, hmm.O, atol=0.005)
    assert jnp.allclose(res_params.mu, hmm.mu, atol=0.000005)

@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_mutli_sequence_multi_mu(mode):
    n_seq = 50

    T = jnp.array([[0.9, 0.1], [0.1,0.9]])
    O = jnp.eye(2)
    mu = jnp.array([[k % 2, (k + 1) % 2] for k in range(n_seq)])
    hmm = HiddenMarkovParameters(T, O, mu)
    init_guess = HiddenMarkovParameters(
        jnp.ones((2,2)) / 2, 
        jnp.ones((2,2)) / 2, 
        jnp.array([[0.49,0.51], [0.51, 0.49]] * (n_seq // 2))
        )

    _, obs = generate_sequence(key(0), hmm, 500)

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
    assert jnp.allclose(res_params.T, hmm.T, atol=0.01)
    assert jnp.allclose(res_params.O, hmm.O, atol=0.005)
    assert jnp.allclose(res_params.mu, hmm.mu, atol=0.000005)

@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_mutli_sequence_multi_mu_informed_parameters(mode):
    n_seq = 50

    T = jnp.array([[0.9, 0.1], [0.1,0.9]])
    O = jnp.eye(2)
    mu = jnp.array([[k % 2, (k + 1) % 2] for k in range(n_seq)])
    hmm = HiddenMarkovParameters(T, O, mu)
    init_guess = HiddenMarkovParameters(
        jnp.array([[0.8, 0.2], [0.2,0.8]]), 
        jnp.array([[0.8, 0.2], [0.2,0.8]]),
        jnp.ones((n_seq, 2)) / 2
        )

    _, obs = generate_sequence(key(0), hmm, 500)

    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
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
    result = baum_welch(reference_obs[:200], init_guess.astype(jnp.float64), max_iter=1000, tol=5e-10, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    # Generate another batch of sequences from the learned parameters 
    _, obs = jax.vmap(lambda _k: generate_sequence(_k, res_params, 100))(jnp.array(seq_keys))
    obs_dist_over_time = jax.lax.map(lambda o: jnp.count_nonzero(obs == o, axis=0), jnp.arange(m)) / n_seq

    assert res_params.mu.shape == (n,)
    assert not result.terminated
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
    assert jnp.allclose(OBS_DISTR_STRUCTURED_100_STEPS, obs_dist_over_time, atol=0.07)
    

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
    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    # Generate another batch of sequences from the learned parameters 
    _, test_obs = jax.vmap(lambda _k: generate_sequence(_k, res_params, 500))(jnp.array(seq_keys))
    mean_obs = jnp.mean(obs)
    mean_test_obs = jnp.mean(test_obs)
    obs_dist = jax.lax.map(lambda o: jnp.mean(obs == o), jnp.arange(m))
    test_obs_dist = jax.lax.map(lambda o: jnp.mean(test_obs == o), jnp.arange(m))

    assert not result.terminated
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
    assert jnp.allclose(mean_test_obs, mean_obs, rtol=0.01)
    assert jnp.allclose(test_obs_dist, obs_dist, rtol=0.02)

@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_structured_learning(mode):
    n_seq = 150
    len_seq = 100
    n, m = O_TEST_STRUCTURED.shape
    key_T, key_O, key_mu, *seq_keys = split(key(345), 3 + n_seq)

    # Prepare structured HMM parameters and sequences
    T_structure = jnp.tril(jnp.triu(jnp.ones_like(T_TEST_STRUCTURED, dtype=jnp.float64)), 2)
    O_structure = jnp.ones_like(O_TEST_STRUCTURED, dtype=jnp.float64)
    O_structure = O_structure.at[n-1, :m-1].set(0)

    _, obs = jax.vmap(
        lambda _k: generate_sequence(_k, HMM_TEST_STRUCTURED.astype(jnp.float64), len_seq)
        )(jnp.array(seq_keys))
    
    # Set up the structured initialization of the HMM parameters
    # with some bias towards the true parameters
    _T = uniform(key_T, (n,n)) * T_structure + jnp.eye(n)
    _O = uniform(key_O, (n,m)) * O_structure + jnp.eye(n,m)
    _mu = uniform(key_mu, n)

    _T = _T / jnp.sum(_T, axis=1)[:, None]
    _O = _O / jnp.sum(_O, axis=1)[:, None]
    _mu = _mu / jnp.sum(_mu)
    init_guess = HiddenMarkovParameters(_T, _O, _mu)


    # Run the algorithm
    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    # Test if parameters were learned correctly 
    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
    assert jnp.allclose(res_params.T, T_TEST_STRUCTURED, atol=0.01)
    assert jnp.allclose(res_params.O, O_TEST_STRUCTURED, atol=0.025)
    assert jnp.allclose(res_params.mu, MU_TEST_STRUCTURED, atol= 0.07)


@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_structured_learning_different_mu(mode):
    n_seq = 100
    len_seq = 100
    n, m = O_TEST_STRUCTURED.shape
    seq_key, param_key = split(key(345))

    # Prepare structured HMM parameters and sequences
    T_structure = jnp.tril(jnp.triu(jnp.ones_like(T_TEST_STRUCTURED, dtype=jnp.float64)), 2)
    O_structure = jnp.ones_like(O_TEST_STRUCTURED, dtype=jnp.float64)
    O_structure = O_structure.at[n-1, :m-1].set(0)
    mu_seq = jnp.concat(
    jnp.array([
        [1.0, 0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], 
        ]
    )[None,...].repeat(n_seq // 4, axis=0))

    _, obs = generate_sequence(
        seq_key, HMM_TEST_STRUCTURED.replace_mu(mu_seq), len_seq)
    
    # Set up the structured initialization of the HMM parameters
    key_T, key_O, key_mu = split(param_key, 3)
    _T = uniform(key_T, (n,n)) * T_structure
    _O = uniform(key_O, (n,m)) * O_structure
    _mu = uniform(key_mu, (n_seq, n))

    _T = _T / jnp.sum(_T, axis=1)[:, None]
    _O = _O / jnp.sum(_O, axis=1)[:, None]
    _mu = _mu / jnp.sum(_mu, axis=1)[:, None]
    init_guess = HiddenMarkovParameters(_T, _O, _mu)


    # Run the algorithm
    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, mode=mode)
    res_params = result.params.to_prob() if mode == 'log' else result.params

    # Test if parameters were learned correctly 
    correct_mu_estimates = jnp.array(
        [
            jnp.allclose(estimate, true_value, atol=0.1) 
            for estimate, true_value in zip(res_params.mu, mu_seq)
        ]
    )
    assert not result.terminated
    assert result.iterations > 5
    assert jnp.all(jnp.diff(result.log_likelihoods[:result.iterations]) >= MONOTONICITY_TOLERANCE)
    assert jnp.allclose(res_params.T, T_TEST_STRUCTURED, atol=0.01)
    assert jnp.allclose(res_params.O, O_TEST_STRUCTURED, atol=0.025)
    assert jnp.mean(correct_mu_estimates) > 0.8 # Correctly identified more than 80% of the mu values


@pytest.mark.slow
@pytest.mark.parametrize('mode', ['log', 'regular'])
@enable_x64
def test_likelihood_lower_bound_increase(mode):
    n_seq = 150
    len_seq = 100
    n, m = O_TEST_STRUCTURED.shape
    key_T, key_O, key_mu, *seq_keys = split(key(345), 3 + n_seq)

    # Prepare structured HMM parameters and sequences
    T_structure = jnp.tril(jnp.triu(jnp.ones_like(T_TEST_STRUCTURED, dtype=jnp.float64)), 2)
    O_structure = jnp.ones_like(O_TEST_STRUCTURED, dtype=jnp.float64)
    O_structure = O_structure.at[n-1, :m-1].set(0)

    _, obs = jax.vmap(
        lambda _k: generate_sequence(_k, HMM_TEST_STRUCTURED.astype(jnp.float64), len_seq)
        )(jnp.array(seq_keys))
    
    # Set up the structured initialization of the HMM parameters
    # with some bias towards the true parameters
    _T = uniform(key_T, (n,n)) * T_structure
    _O = uniform(key_O, (n,m)) * O_structure
    _mu = uniform(key_mu, n)

    _T = _T / jnp.sum(_T, axis=1)[:, None]
    _O = _O / jnp.sum(_O, axis=1)[:, None]
    _mu = _mu / jnp.sum(_mu)
    init_guess = HiddenMarkovParameters(_T, _O, _mu)


    # Run the algorithm
    result = baum_welch(obs, init_guess.astype(jnp.float64), max_iter=500, tol=1e-6, check_ascent=True, mode=mode)

    # Test if likelihood trend is increasing: 
    averaged_increases = jnp.diff(
        jnp.convolve(
            result.log_likelihoods[:result.iterations], 
            jnp.ones(5) / 5, 
            mode='valid'
            )
        )
    assert not result.terminated
    assert result.iterations > 6
    assert jnp.all(averaged_increases >= MONOTONICITY_TOLERANCE), ',\n '.join(map(str, averaged_increases.tolist()))


@pytest.mark.debug
@pytest.mark.parametrize('mode', ['regular', 'log'])
@enable_x64
def test_multiple_sequence_8_states(mode):
    
    result = baum_welch(
        TEST_SEQUENCES_REAL_DATA_8_STATES, 
        HMM_TEST_REAL_DATA.astype(jnp.float64), 
        1000,
        mode=mode)

    assert not jnp.any(jnp.isnan(result.params.mu)), jnp.exp(result.params.mu)
    assert not jnp.any(jnp.isnan(result.params.O)), result.params.O
    assert not jnp.any(jnp.isnan(result.params.T)), jnp.exp(result.params.T)
    assert not result.terminated

# Make sure that freezing of parameters works as intendedG
@pytest.mark.debug
@pytest.mark.parametrize('mode', ['regular', 'log'])
@pytest.mark.parametrize('freeze_config', [
    FreezeConfig(T=True),
    FreezeConfig(O=True),
    FreezeConfig(mu=True),
])
@enable_x64
def test_freeze_config(mode, freeze_config):
    n, m = HMM_TEST.O.shape
    hmm = HiddenMarkovParameters(
        normalize_rows(jax.random.uniform(key(0), (n, n))), 
        normalize_rows(jax.random.uniform(key(1), (n, m))), 
        normalize_rows(jax.random.uniform(key(2), (n, ))),
        is_log = (mode=='log')).astype(jnp.float64)
    
    _result = baum_welch(
        TEST_SEQUENCES_5_STEPS[0], 
        hmm, 
        max_iter=5, 
        mode=mode, 
        freeze_config=freeze_config)

    result = _result.params
    if freeze_config.T:
        assert jnp.allclose(result.T, hmm.T)
    else:
        assert not jnp.allclose(result.T, hmm.T)
    if freeze_config.O:
        assert jnp.allclose(result.O, hmm.O)
    else:
        assert not jnp.allclose(result.O, hmm.O)
    if freeze_config.mu:
        assert jnp.allclose(result.mu, hmm.mu)
    else:
        assert not jnp.allclose(result.mu, hmm.mu)