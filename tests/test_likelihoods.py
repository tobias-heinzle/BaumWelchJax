import jax.numpy as jnp
from jax.scipy.special import logsumexp

import pytest

from baum_welch_jax.algorithms import likelihood, log_likelihood
from baum_welch_jax.models import HiddenMarkovParameters

from conftest import *


# Test with trivial parameters for a sanity check
@pytest.mark.parametrize(
        'initial_state, obs, likelihoods', 
        [
            (0, jnp.zeros(1000).astype(jnp.int32), 1.0),
            (0, jnp.ones(1000).astype(jnp.int32), 0.0),
            (1, jnp.zeros(1000).astype(jnp.int32), 0.0),
            (1, jnp.ones(1000).astype(jnp.int32), 1.0),
        ]
    )
def test_likelihood_sanity_check(initial_state, obs, likelihoods):
    T = jnp.eye(2)
    O = jnp.eye(2)
    mu = jnp.zeros(2)
    mu = mu.at[initial_state].set(1.0)
    hmm = HiddenMarkovParameters(T, O, mu)

    state_llhood, llhood_sequence = likelihood(obs, hmm, return_stats=True)
    llhood = likelihood(obs, hmm, return_stats=False)

    assert len(llhood_sequence) == 1000
    assert state_llhood.shape == (2,)
    assert jnp.allclose(llhood_sequence, likelihoods)
    assert jnp.allclose(llhood, likelihoods)

@pytest.mark.parametrize(
        'initial_state, obs, likelihoods', 
        [
            (0, jnp.zeros(1000).astype(jnp.int32), 1.0),
            (0, jnp.ones(1000).astype(jnp.int32), 0.0),
            (1, jnp.zeros(1000).astype(jnp.int32), 0.0),
            (1, jnp.ones(1000).astype(jnp.int32), 1.0),
        ]
    )
def test_log_likelihood_sanity_check(initial_state, obs, likelihoods):
    T = jnp.eye(2)
    O = jnp.eye(2)
    mu = jnp.zeros(2)
    mu = mu.at[initial_state].set(1.0)
    hmm = HiddenMarkovParameters(T, O, mu)

    state_log_llhood, log_llhood_sequence = log_likelihood(obs, hmm, return_stats=True)
    log_llhood = log_likelihood(obs, hmm, return_stats=False)

    assert len(log_llhood_sequence) == 1000
    assert state_log_llhood.shape == (2,)
    assert jnp.allclose(jnp.exp(log_llhood_sequence), likelihoods)
    assert jnp.allclose(jnp.exp(log_llhood), likelihoods)

def test_likelihoods_dtype_errors():
    hmm_log = HMM_TEST.to_log()
    hmm = hmm_log.to_prob()
    obs = jnp.zeros(100).astype(jnp.float32)

    with pytest.raises(ValueError):
        likelihood(obs, hmm)

    with pytest.raises(ValueError):
        log_likelihood(obs, hmm_log)

def test_likelihoods_obs_mu_mismatch():
    hmm = HiddenMarkovParameters(jnp.eye(2), jnp.eye(2), jnp.eye(2))
    hmm_log = hmm.to_log()
    obs = jnp.zeros(100).astype(jnp.int32)

    with pytest.raises(ValueError):
        likelihood(obs, hmm)

    with pytest.raises(ValueError):
        log_likelihood(obs, hmm_log)


@pytest.mark.parametrize('eps, n', [(1e-6, 1000), (1e-6, 10_000), (1e-10, 1000), (1e-10, 10_000)])
def test_log_likelihood_long_sequence(eps, n):
    T = jnp.array([[eps, 1 - eps], [0.0, 1.0]])
    O = jnp.eye(2)
    mu = jnp.array([1.0, 0.0])
    hmm = HiddenMarkovParameters(T, O, mu)
    obs = jnp.zeros(n)
    obs = jnp.astype(obs, jnp.int32)

    log_llhood = log_likelihood(obs, hmm, return_stats=False)

    assert jnp.allclose(log_llhood, jnp.log(eps) * (n - 1), rtol=0.0001)

# Test with random parameters
@pytest.mark.parametrize('obs, llhood, final_state_distr', 
                         zip(TEST_SEQUENCES_5_STEPS, TEST_LIKELIHOODS_5_STEPS, FINAL_STATE_DISTR_5_STEPS))
def test_likelihood_sampled_short(obs, llhood, final_state_distr):
    state_llhoods, llhood_seq = likelihood(obs, HMM_TEST, return_stats=True)
    state_distribution = state_llhoods / jnp.sum(state_llhoods)

    assert jnp.allclose(llhood_seq[-1], llhood, rtol=0.03)
    assert jnp.allclose(state_distribution, final_state_distr, rtol=0.07)


@pytest.mark.parametrize('obs, llhood, final_state_distr', 
                         zip(TEST_SEQUENCES_5_STEPS, TEST_LIKELIHOODS_5_STEPS, FINAL_STATE_DISTR_5_STEPS))
def test_log_likelihood_sampled_short(obs, llhood, final_state_distr):
    state_log_llhoods, log_llhood_seq = log_likelihood(obs, HMM_TEST, return_stats=True)
    state_distribution = jnp.exp(state_log_llhoods - logsumexp(state_log_llhoods))
    log_llhood = log_llhood_seq[-1]

    assert jnp.allclose(jnp.exp(log_llhood), llhood, rtol=0.03)
    assert jnp.allclose(state_distribution, final_state_distr, rtol=0.07)

@pytest.mark.parametrize('obs, llhood', zip(TEST_SEQUENCES_7_STEPS, TEST_LIKELIHOODS_7_STEPS))
def test_likelihood_sampled_long(obs, llhood):
    computed_llhood = likelihood(obs, HMM_TEST, return_stats=False)

    assert jnp.allclose(computed_llhood, llhood, rtol=0.05)


@pytest.mark.parametrize('obs, llhood', zip(TEST_SEQUENCES_7_STEPS, TEST_LIKELIHOODS_7_STEPS))
def test_log_likelihood_sampled_long(obs, llhood):
    computed_log_llhood = log_likelihood(obs, HMM_TEST, return_stats=False)

    assert jnp.allclose(jnp.exp(computed_log_llhood), llhood, rtol=0.05)

def test_likelihood_multiple_obs_single_mu():
    obs_arr = jnp.array(TEST_SEQUENCES_5_STEPS)
    test_llhoods = jnp.array(TEST_LIKELIHOODS_5_STEPS)
    final_state_distr = jnp.array(FINAL_STATE_DISTR_5_STEPS)

    state_llhoods, llhood_seq = likelihood(obs_arr, HMM_TEST, return_stats=True)
    state_distributions = state_llhoods / jnp.sum(state_llhoods, axis=-1)[..., None]

    assert jnp.allclose(llhood_seq[:, -1], test_llhoods, rtol=0.03)
    assert jnp.allclose(state_distributions, final_state_distr, rtol=0.07)

def test_log_likelihood_multiple_obs_single_mu():
    obs_arr = jnp.array(TEST_SEQUENCES_5_STEPS)
    test_llhoods = jnp.array(TEST_LIKELIHOODS_5_STEPS)
    final_state_distr = jnp.array(FINAL_STATE_DISTR_5_STEPS)

    state_llhoods, llhood_seq = log_likelihood(obs_arr, HMM_TEST.to_log(), return_stats=True)
    state_distributions = jnp.exp(state_llhoods - logsumexp(state_llhoods, axis=-1)[..., None])

    assert jnp.allclose(jnp.exp(llhood_seq[:, -1]), test_llhoods, rtol=0.03)
    assert jnp.allclose(state_distributions, final_state_distr, rtol=0.07)



def test_likelihood_multiple_obs_multiple_mu():
    # This test checks if the correct likelihood and final state distributions are
    # computed if different initial state distributions are used for each sequence!
    obs_arr = jnp.array(TEST_SEQUENCES_5_STEPS + TEST_SEQUENCES_5_STEPS_DIFFERENT_MU)
    test_llhoods = jnp.array(TEST_LIKELIHOODS_5_STEPS + TEST_LIKELIHOODS_5_STEPS_DIFFERENT_MU)
    final_state_distr = jnp.array(FINAL_STATE_DISTR_5_STEPS + FINAL_STATE_DISTR_5_STEPS_DIFFERENT_MU)

    k = len(TEST_SEQUENCES_5_STEPS)
    l = len(TEST_SEQUENCES_5_STEPS_DIFFERENT_MU)

    hmm = HiddenMarkovParameters(T_TEST, O_TEST, jnp.array([MU_TEST]*k + [MU_TEST_DIFFERENT]*l))

    state_llhoods, llhood_seq = likelihood(obs_arr, hmm, return_stats=True)
    state_distributions = state_llhoods / jnp.sum(state_llhoods, axis=-1)[..., None]

    assert jnp.allclose(llhood_seq[:, -1], test_llhoods, rtol=0.03)
    assert jnp.allclose(state_distributions, final_state_distr, rtol=0.09)

def test_log_likelihood_multiple_obs_multiple_mu():
    # This test checks if the correct log likelihood and final state distributions are
    # computed if different initial state distributions are used for each sequence!
    obs_arr = jnp.array(TEST_SEQUENCES_5_STEPS + TEST_SEQUENCES_5_STEPS_DIFFERENT_MU)
    test_llhoods = jnp.array(TEST_LIKELIHOODS_5_STEPS + TEST_LIKELIHOODS_5_STEPS_DIFFERENT_MU)
    final_state_distr = jnp.array(FINAL_STATE_DISTR_5_STEPS + FINAL_STATE_DISTR_5_STEPS_DIFFERENT_MU)

    k = len(TEST_SEQUENCES_5_STEPS)
    l = len(TEST_SEQUENCES_5_STEPS_DIFFERENT_MU)

    hmm = HiddenMarkovParameters(T_TEST, O_TEST, jnp.array([MU_TEST]*k + [MU_TEST_DIFFERENT]*l))

    state_llhoods, llhood_seq = log_likelihood(obs_arr, hmm.to_log(), return_stats=True)
    state_distributions = jnp.exp(state_llhoods - logsumexp(state_llhoods, axis=-1)[..., None])

    assert jnp.allclose(jnp.exp(llhood_seq[:, -1]), test_llhoods, rtol=0.03)
    assert jnp.allclose(state_distributions, final_state_distr, rtol=0.09)


# Test with structured parameters
@pytest.mark.parametrize('obs, llhood, final_state_distr', 
                         zip(
                             TEST_SEQUENCES_STRUCTURED_5_STEPS, 
                             TEST_LIKELIHOODS_STRUCTURED_5_STEPS, 
                             FINAL_STATE_DISTR_STRUCTURED_5_STEPS
                             )
                        )
def test_likelihood_structured_sampled_short(obs, llhood, final_state_distr):
    state_llhoods, llhood_seq = likelihood(
        obs, HMM_TEST_STRUCTURED, return_stats=True)
    state_distribution = state_llhoods / jnp.sum(state_llhoods)

    assert jnp.allclose(llhood_seq[-1], llhood, rtol=0.03)
    assert jnp.allclose(state_distribution, final_state_distr, rtol=0.07)


@pytest.mark.parametrize('obs, llhood, final_state_distr', 
                          zip(
                             TEST_SEQUENCES_STRUCTURED_5_STEPS, 
                             TEST_LIKELIHOODS_STRUCTURED_5_STEPS, 
                             FINAL_STATE_DISTR_STRUCTURED_5_STEPS
                             )
                        )
def test_log_likelihood_structured_sampled_short(obs, llhood, final_state_distr):
    state_log_llhoods, log_llhood_seq = log_likelihood(
        obs, HMM_TEST_STRUCTURED, return_stats=True)
    state_distribution = jnp.exp(state_log_llhoods - logsumexp(state_log_llhoods))
    log_llhood = log_llhood_seq[-1]

    assert jnp.allclose(jnp.exp(log_llhood), llhood, rtol=0.03)
    assert jnp.allclose(state_distribution, final_state_distr, rtol=0.07)

@pytest.mark.parametrize('obs, llhood', 
                          zip(
                             TEST_SEQUENCES_STRUCTURED_6_STEPS, 
                             TEST_LIKELIHOODS_STRUCTURED_6_STEPS, 
                             )
                        )
def test_likelihood_structured_sampled_long(obs, llhood):
    computed_llhood = likelihood(
        obs, HMM_TEST_STRUCTURED, return_stats=False)

    assert jnp.allclose(computed_llhood, llhood, rtol=0.05)


@pytest.mark.parametrize('obs, llhood', 
                          zip(
                             TEST_SEQUENCES_STRUCTURED_6_STEPS, 
                             TEST_LIKELIHOODS_STRUCTURED_6_STEPS, 
                             )
                        )
def test_log_likelihood_structured_sampled_long(obs, llhood):
    computed_log_llhood = log_likelihood(obs, HMM_TEST_STRUCTURED, return_stats=False)

    assert jnp.allclose(jnp.exp(computed_log_llhood), llhood, rtol=0.05)

# Test if both algorithms compute the same numbers for relatively short sequences
@pytest.mark.parametrize('obs', TEST_SEQUENCES_5_STEPS + TEST_SEQUENCES_6_STEPS + TEST_SEQUENCES_7_STEPS)
def test_likelihood_agreement_structured(obs):
    log_llhood = log_likelihood(obs, HMM_TEST, return_stats=False)
    llhood = likelihood(obs, HMM_TEST, return_stats=False)

    assert jnp.allclose(llhood, jnp.exp(log_llhood))


@pytest.mark.parametrize('obs', TEST_SEQUENCES_STRUCTURED_5_STEPS + TEST_SEQUENCES_STRUCTURED_6_STEPS)
def test_likelihood_agreement_structured(obs):
    log_llhood = log_likelihood(obs, HMM_TEST_STRUCTURED, return_stats=False)
    llhood = likelihood(obs, HMM_TEST_STRUCTURED, return_stats=False)

    assert jnp.allclose(llhood, jnp.exp(log_llhood))
