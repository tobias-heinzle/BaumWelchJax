import jax
import jax.numpy as jnp

import pytest

from baum_welch_jax.algorithms import forward_backward

from conftest import *

@pytest.mark.parametrize('mode', ['regular', 'log'])
def test_fwd_bwd_dimensions(mode):
	obs = jnp.zeros(100).astype(jnp.int32)
	gamma, xi = forward_backward(obs, HMM_TRIVIAL, mode=mode)

	assert gamma.shape[0] == 100
	assert xi.shape[0] == 99


def test_fwd_bwd_dtype_errors():
	obs = jnp.zeros(100).astype(jnp.float32)
	
	with pytest.raises(ValueError):
		forward_backward(obs, HMM_TRIVIAL)

@pytest.mark.parametrize('mode', ['regular', 'log'])
@pytest.mark.parametrize('hmm', [HMM_TRIVIAL, HMM_TRIVIAL.to_log()])
def test_fwd_bwd_trivial(hmm, mode):
	obs = jnp.zeros(10).astype(jnp.int32)
	state_distr = jnp.zeros((10,5))
	state_distr = state_distr.at[:,0].set(1.0)
	trans_distr = jnp.zeros((9,5,5))
	trans_distr = trans_distr.at[:,0,0].set(1.0)
	if mode == 'log':
		state_distr = jnp.log(state_distr)
		trans_distr = jnp.log(trans_distr)
	
	gamma, xi = forward_backward(obs, hmm, mode=mode)

	assert jnp.allclose(gamma, state_distr)
	assert jnp.allclose(xi, trans_distr)

	


# Test the gamma computation with random parameter set
@pytest.mark.parametrize('obs, sampled_distr', zip(TEST_SEQUENCES_5_STEPS, STATE_DISTR_5_STEPS))
def test_fwd_bwd_gamma_5_steps(obs, sampled_distr):
	gamma, _ = forward_backward(obs, HMM_TEST, mode='regular')
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.02)

@pytest.mark.parametrize('obs, sampled_distr', zip(TEST_SEQUENCES_5_STEPS, STATE_DISTR_5_STEPS))
def test_fwd_bwd_gamma_log_5_steps(obs, sampled_distr):
	gamma_log, _ = forward_backward(obs, HMM_TEST, mode='log')
	gamma = jnp.exp(gamma_log)
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.02)

@pytest.mark.parametrize('obs, sampled_distr', zip(TEST_SEQUENCES_6_STEPS, STATE_DISTR_6_STEPS))
def test_fwd_bwd_gamma_6_steps(obs, sampled_distr):
	gamma, _ = forward_backward(obs, HMM_TEST, mode='regular')
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.04)

@pytest.mark.parametrize('obs, sampled_distr', zip(TEST_SEQUENCES_6_STEPS, STATE_DISTR_6_STEPS))
def test_fwd_bwd_gamma_log_6_steps(obs, sampled_distr):
	gamma_log, _ = forward_backward(obs, HMM_TEST, mode='log')
	gamma = jnp.exp(gamma_log)
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.04)
	

# Test the gamma computation with structured parameter set (Band structure of T, zero elements also in O)
@pytest.mark.parametrize('obs, sampled_distr', 
						 zip(TEST_SEQUENCES_STRUCTURED_5_STEPS, STATE_DISTR_STRUCTURED_5_STEPS))
def test_fwd_bwd_structured_gamma_5_steps(obs, sampled_distr):
	gamma, _ = forward_backward(
		obs, HMM_TEST_STRUCTURED, mode='regular')
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.02)

@pytest.mark.parametrize(
		'obs, sampled_distr', zip(TEST_SEQUENCES_STRUCTURED_5_STEPS, STATE_DISTR_STRUCTURED_5_STEPS))
def test_fwd_bwd_structured_gamma_log_5_steps(obs, sampled_distr):
	gamma_log, _ = forward_backward(
		obs, HMM_TEST_STRUCTURED, mode='log')
	gamma = jnp.exp(gamma_log)
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.02)

@pytest.mark.parametrize(
		'obs, sampled_distr', zip(TEST_SEQUENCES_STRUCTURED_6_STEPS, STATE_DISTR_STRUCTURED_6_STEPS))
def test_fwd_bwd_structured_gamma_6_steps(obs, sampled_distr):
	gamma, _ = forward_backward(
		obs, HMM_TEST_STRUCTURED, mode='regular')
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.04)

@pytest.mark.parametrize(
		'obs, sampled_distr', zip(TEST_SEQUENCES_STRUCTURED_6_STEPS, STATE_DISTR_STRUCTURED_6_STEPS))
def test_fwd_bwd_structured_gamma_log_6_steps(obs, sampled_distr):
	gamma_log, _ = forward_backward(
		obs, HMM_TEST_STRUCTURED, mode='log')
	gamma = jnp.exp(gamma_log)
	
	assert jnp.allclose(gamma.T, sampled_distr, atol=0.04)


# Test xi computation on random parameter set
@pytest.mark.parametrize('obs, sampled_distr', zip(TEST_SEQUENCES_6_STEPS, TRANSITION_TENSORS_TEST_6_STEPS))
def test_fwd_bwd_xi_5_steps(obs, sampled_distr):
	_, xi = forward_backward(obs, HMM_TEST, mode='regular')
	
	assert jnp.allclose(xi[0], sampled_distr[0], atol=0.015)
	assert jnp.allclose(xi[1], sampled_distr[1], atol=0.02)
	assert jnp.allclose(xi[2], sampled_distr[2], atol=0.03)
	assert jnp.allclose(xi[3], sampled_distr[3], atol=0.05)
	assert jnp.allclose(xi[4], sampled_distr[4], atol=0.11)
	

@pytest.mark.parametrize('obs, sampled_distr', zip(TEST_SEQUENCES_6_STEPS, TRANSITION_TENSORS_TEST_6_STEPS))
def test_fwd_bwd_xi_log_5_steps(obs, sampled_distr):
	_, xi_log = forward_backward(obs, HMM_TEST, mode='log')
	xi = jnp.exp(xi_log)
	
	assert jnp.allclose(xi[0], sampled_distr[0], atol=0.015)
	assert jnp.allclose(xi[1], sampled_distr[1], atol=0.02)
	assert jnp.allclose(xi[2], sampled_distr[2], atol=0.03)
	assert jnp.allclose(xi[3], sampled_distr[3], atol=0.05)
	assert jnp.allclose(xi[4], sampled_distr[4], atol=0.11)
	

# Test xi computation with structured parameter set (Band structure of T, zero elements also in O)
@pytest.mark.parametrize(
		'obs, sampled_distr', zip(TEST_SEQUENCES_STRUCTURED_6_STEPS, TRANSITION_TENSORS_STRUCTURED_TEST_6_STEPS))
def test_fwd_bwd_structured_xi_5_steps(obs, sampled_distr):
	_, xi = forward_backward(obs, HMM_TEST_STRUCTURED, mode='regular')
	
	assert jnp.allclose(xi[0], sampled_distr[0], atol=0.015)
	assert jnp.allclose(xi[1], sampled_distr[1], atol=0.02)
	assert jnp.allclose(xi[2], sampled_distr[2], atol=0.03)
	assert jnp.allclose(xi[3], sampled_distr[3], atol=0.05)
	assert jnp.allclose(xi[4], sampled_distr[4], atol=0.11)
	

@pytest.mark.parametrize(
		'obs, sampled_distr', zip(TEST_SEQUENCES_STRUCTURED_6_STEPS, TRANSITION_TENSORS_STRUCTURED_TEST_6_STEPS))
def test_fwd_bwd_structured_xi_log_5_steps(obs, sampled_distr):
	_, xi_log = forward_backward(obs, HMM_TEST_STRUCTURED, mode='log')
	xi = jnp.exp(xi_log)
	
	assert jnp.allclose(xi[0], sampled_distr[0], atol=0.015)
	assert jnp.allclose(xi[1], sampled_distr[1], atol=0.02)
	assert jnp.allclose(xi[2], sampled_distr[2], atol=0.03)
	assert jnp.allclose(xi[3], sampled_distr[3], atol=0.05)
	assert jnp.allclose(xi[4], sampled_distr[4], atol=0.11)
	

# Test if both versions of the algorithm arrive at the same result for relatively short sequences
@pytest.mark.parametrize('obs', 
						 TEST_SEQUENCES_5_STEPS 
						 + TEST_SEQUENCES_6_STEPS
						 + TEST_SEQUENCES_7_STEPS)
def test_fwd_bwd_agreement(obs):
	gamma, xi = forward_backward(obs, HMM_TEST, mode='regular')
	gamma_log, xi_log = forward_backward(obs, HMM_TEST, mode='log')
	
	assert jnp.allclose(gamma, jnp.exp(gamma_log))
	assert jnp.allclose(xi, jnp.exp(xi_log))


@pytest.mark.parametrize('obs', 
						 TEST_SEQUENCES_STRUCTURED_5_STEPS
						 + TEST_SEQUENCES_STRUCTURED_6_STEPS)
def test_fwd_bwd_agreement_structured(obs):
	gamma, xi = forward_backward(obs, HMM_TEST_STRUCTURED, mode='regular')
	gamma_log, xi_log = forward_backward(obs, HMM_TEST_STRUCTURED, mode='log')
	
	assert jnp.allclose(gamma, jnp.exp(gamma_log))
	assert jnp.allclose(xi, jnp.exp(xi_log))
