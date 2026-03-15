import functools

import jax
from jax.random import key, split, uniform
import jax.numpy as jnp

import pytest

# from baum_welch_jax import PrecisionWarning
# from baum_welch_jax.algorithms import baum_welch, generate_sequence
# from baum_welch_jax.models import HiddenMarkovParameters, assert_valid_hmm, FreezeConfig, FreezeMasks
# from baum_welch_jax.util import normalize_rows
from baum_welch_jax.models.observation_models import DiscreteObservationModel

from conftest import *



@pytest.mark.parametrize('is_log', [True, False])
def test_discrete_basic_functionality(is_log):

    O_mat = jnp.eye(2)
    if is_log:
        O_mat = jnp.log(O_mat)
    
    O = DiscreteObservationModel(O_mat, is_log=is_log)

    assert jnp.allclose(O.obs_cdf(0), 1.0)
    assert jnp.allclose(O.obs_cdf(1), jnp.array([0.0, 1.0]))
    assert jnp.allclose(O.llhood(0), jnp.array([1.0, 0.0]))
    assert jnp.allclose(O.llhood(1), jnp.array([0.0, 1.0]))
    assert jnp.allclose(O.logllihood(0), jnp.array([0.0, -jnp.inf]))
    assert jnp.allclose(O.logllihood(1), jnp.array([-jnp.inf, 0.0]))


@pytest.mark.parametrize('is_log', [True, False])
def test_discrete_simulate(is_log):

    O_mat = jnp.array([[0.5, 0.5], [0.2, 0.8]])
    if is_log:
        O_mat = jnp.log(O_mat)
    
    O = DiscreteObservationModel(O_mat, is_log=is_log)

    assert jnp.allclose(O.simulate(0, 1.0), 1.0)
    assert jnp.allclose(O.simulate(0, 0.0), 0.0)
    assert jnp.allclose(O.simulate(1, 0.3), 1.0)
    assert jnp.allclose(O.simulate(1, 0.1), 0.0)


def test_illegal_type_conversion():
    O = DiscreteObservationModel(jnp.eye(2), False)

    with pytest.raises(ValueError):
        O.astype(jnp.int32)


@pytest.mark.parametrize('is_log', [True, False])
def test_discrete_update(is_log):
    # TODO: implement this using the generate and forward backward functions
    #       An updated parameter set should be closer to the true parameters.
    raise NotImplementedError
