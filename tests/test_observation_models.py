import jax
import jax.numpy as jnp

import pytest

from baum_welch_jax.algorithms import generate_sequence, forward_backward
from baum_welch_jax.models import HiddenMarkovParameters
from baum_welch_jax.models.observation_models import DiscreteModel, MultivariateGaussianModel, ObservationModel

from conftest import *



@pytest.mark.parametrize('is_log', [True, False])
def test_discrete_basic_functionality(is_log):

    O_mat = jnp.eye(2)
    if is_log:
        O_mat = jnp.log(O_mat)
    
    O = DiscreteModel(O_mat, is_log=is_log)

    assert jnp.allclose(O.llhood(0), jnp.array([1.0, 0.0]))
    assert jnp.allclose(O.llhood(1), jnp.array([0.0, 1.0]))
    assert jnp.allclose(O.logllhood(0), jnp.array([0.0, -jnp.inf]))
    assert jnp.allclose(O.logllhood(1), jnp.array([-jnp.inf, 0.0]))


@pytest.mark.parametrize('is_log', [True, False])
def test_discrete_simulate(is_log):

    O_mat = jnp.array([[0.5, 0.5], [0.2, 0.8]])
    if is_log:
        O_mat = jnp.log(O_mat)
    
    O = DiscreteModel(O_mat, is_log=is_log)

    assert jnp.allclose(O.simulate(0, 1.0), 1.0)
    assert jnp.allclose(O.simulate(0, 0.0), 0.0)
    assert jnp.allclose(O.simulate(1, 0.3), 1.0)
    assert jnp.allclose(O.simulate(1, 0.1), 0.0)


def test_illegal_type_conversion():
    O = DiscreteModel(jnp.eye(2), False)

    with pytest.raises(ValueError):
        O.astype(jnp.int32)


OBSERVATION_MODELS = [
    (DiscreteModel(jnp.array([[0.95, 0.05], [0.05, 0.95]])), DiscreteModel(jnp.full((2,2), 1/2))),
    (
        MultivariateGaussianModel(
            mean=jnp.array([[1.0, 1.0], [-1.0, -1.0]]), 
            covariance=jnp.eye(2)[None, ...].repeat(2, axis=0)
            ),
        MultivariateGaussianModel(
            mean=jnp.zeros((2,2)),
            covariance=jnp.full((2,2,2), 1.5) # Not valid, but never used
        )
    )
]

def diff(x, y):
    return jnp.abs(x - y)

@pytest.mark.parametrize('true_model, test_model', OBSERVATION_MODELS)
@pytest.mark.parametrize('is_log', [True, False])
def test_update(true_model: ObservationModel, test_model: ObservationModel, is_log: bool):
    T = jnp.array([[0.4, 0.6], [0.9, 0.1]])
    mu = jnp.array([[0.0, 1.0]] * 100)
    hmm = HiddenMarkovParameters(T, true_model, mu)

    if is_log:
        hmm = hmm.to_log()
        true_model = true_model.to_log()
        test_model = test_model.to_log()

    # Because we are generating the gamma tensor with the true parameters, 
    # the update should produce a new observation model very close to the 
    # ground truth
    _, obs = generate_sequence(jax.random.key(0), hmm, 50)    
    gamma, _ = forward_backward(obs, hmm, 'log' if is_log else 'regular', squeeze=False)
    gamma = jnp.concat(gamma, axis=0)

    new_obs_model = test_model.update(obs, gamma)

    # Check if the difference to the true parameter has been reduced
    initial_diff = jax.tree.map(diff, true_model, test_model)
    new_diff = jax.tree.map(diff, true_model, new_obs_model)
    is_less = jax.tree.map(jnp.less, new_diff, initial_diff)
    all_less = jax.tree.reduce(lambda r, x: r and jnp.all(x), is_less, True)

    assert all_less, f'Update did not improve parameters: \n{new_obs_model}'
