from dataclasses import dataclass, field
from typing import Self
from abc import ABC, abstractmethod

import jax
from jax import Array, lax
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal

import jax.numpy as jnp

class ObservationModel(ABC):

    @abstractmethod
    def update(self, obs: Array, gamma: Array) -> Self:
        '''Compute an updated version of the observation parameters based on the
        sufficient statistics gamma and the observation sequence obs.'''
        raise NotImplementedError

    @abstractmethod
    def llhood(self, obs: Array) -> Array:
        '''Unnormalized likelihood of the states, given the observation.'''
        raise NotImplementedError

    @abstractmethod
    def logllhood(self, obs: Array) -> Array:
        '''Unnormalized log likelihood of the states, given the observation.'''
        raise NotImplementedError

    @abstractmethod
    def simulate(self, state: Array, uniform_sample: Array) -> Array:
        '''Transform a sample of a uniform distribution into a sample of the observation
        distribution.'''
        raise NotImplementedError

    @abstractmethod
    def to_log(self) -> Self:
        '''Convert parameters to log params'''
        raise NotImplementedError

    @abstractmethod
    def to_prob(self) -> Self:
        '''Convert parameters from log params to regular probabilities'''
        raise NotImplementedError
    
    @abstractmethod
    def astype(self, dtype) -> Self:
        '''Convert arrays to the provided dtype'''
        raise NotImplementedError
    
    @abstractmethod
    def construct_frozen_parameter_pytree(self, mask: Array) -> Self:
        '''Construct a pytree suitable for mapped masking of parameters.'''
        raise NotImplementedError
    
    @abstractmethod
    def get_params(self) -> Array:
        raise NotImplementedError
    
    @abstractmethod
    def squeeze(self) -> Self:
        raise NotImplementedError
    
    @abstractmethod
    def check_obs_compatibility(self, obs) -> bool:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def is_valid(self) -> bool:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def ndim(self) -> int:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def dtype(self) -> jax.typing.DTypeLike:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def has_multiple_outputs(self) -> bool:
        raise NotImplementedError
    




@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscreteModel(ObservationModel):
    obs_probs: Array
    is_log: bool = field(metadata={"static": True}, default=False)

    def update(self, obs: Array, gamma: Array) -> Self:

        m = self.obs_probs.shape[-1]

        if self.is_log:
            O = lax.map(lambda o: logsumexp(
                # This log will be -inf at a given index if obs != o there!
                jnp.log(obs.ravel() == o)[:, None] + gamma, axis=0), jnp.arange(m)).T
            O -= logsumexp(gamma, axis=0)[..., None]
        else:    
            O = lax.map(lambda o: jnp.sum(
                (obs.ravel() == o)[:, None] * gamma, axis=0), jnp.arange(m)).T
            O = O / jnp.sum(gamma, axis=0)[..., None]

        return DiscreteModel(O, self.is_log)


    def llhood(self, obs: Array) -> Array:
        if self.is_log:
            return jnp.exp(self.obs_probs[:, obs])
        
        return self.obs_probs[:, obs]
    
    def logllhood(self, obs: Array) -> Array:
        if self.is_log:
            return self.obs_probs[:, obs]

        return jnp.log(self.obs_probs[:, obs])
    
    def _obs_cdf(self, state):
        # TODO: should this be field of the class, instead of computation 
        # for every sample?
        if self.is_log:
            return jnp.cumsum(jnp.exp(self.obs_probs[state]))
        
        return jnp.cumsum(self.obs_probs[state])
    
    def simulate(self, state: Array, uniform_sample: Array) -> Array:
        return jnp.argmax(self._obs_cdf(state) >= uniform_sample)
    
    def to_log(self) -> Self:
        if self.is_log:
            raise ValueError('Attempted log conversion; Parameters are already logprobabilities.')
        return DiscreteModel(jnp.log(self.obs_probs), True)
    
    def to_prob(self) -> Self:
        if self.is_log:
            return DiscreteModel(jnp.exp(self.obs_probs), False)
        raise ValueError('Attempted probability conversion; Parameters are already probabilities.')
    
    def astype(self, dtype: jax.typing.DTypeLike) -> Self:
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("dtype must be floating point number")
        
        return DiscreteModel(
            self.obs_probs.astype(dtype),
            self.is_log
        )
    
    def construct_frozen_parameter_pytree(self, mask: Array) -> Self:
        return DiscreteModel(mask, self.is_log)
    
    def get_params(self) -> Array:
        return self.obs_probs
    
    def squeeze(self) -> Self:
        return DiscreteModel(self.obs_probs.squeeze(), self.is_log)

    def check_obs_compatibility(self, obs: Array) -> bool:
        return jnp.issubdtype(obs.dtype, jnp.integer)

    @property
    def ndim(self) -> int:
        return self.obs_probs.ndim
    
    @property
    def dtype(self) -> jax.typing.DTypeLike:
        return self.obs_probs.dtype
    
    @property
    def is_valid(self) -> bool:
        correct_dims = self.ndim == 2
        is_float = jnp.issubdtype(self.dtype, jnp.floating)
        if self.is_log:
            all_sum_to_one = jnp.allclose(logsumexp(self.obs_probs, axis=1), 0.0, atol=1e-6)

            return jnp.all(jnp.array([correct_dims, all_sum_to_one]))

        else:
            all_positive = jnp.all(self.obs_probs >= 0)
            all_sum_to_one = jnp.allclose(jnp.sum(self.obs_probs, axis=1), 1.0, atol=1e-6)
              
            return jnp.all(jnp.array([correct_dims, all_positive, all_sum_to_one, is_float]))
        
    @property
    def has_multiple_outputs(self) -> bool:
        return False
    
    def __str__(self):
        return f'''Discrete observation model(\n\tobs_probs = \n{self.obs_probs}\n\n\tis_log = {self.is_log}\n)'''
    


# TODO: Test this and add some FWD BWD and BW tests using this as well

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MultivariateGaussianModel(ObservationModel):
    mean: Array
    covariance: Array
    is_log: bool = field(metadata={"static": True}, default=False)

    def update(self, obs: Array, gamma: Array) -> Self:
        assert obs.ndim == 3, f'obs.ndim = {obs.ndim} != 3'

        obs = jnp.concat(obs, axis=0)


        if self.is_log:
            # If the is_log flag is set, we expect a gamma tensor
            # with log values!
            gamma = jnp.exp(gamma)

        norm_const = jnp.sum(gamma, axis=0)

        mean = jnp.einsum('ij, ik -> jk', gamma, obs)
        mean = mean / norm_const[:, None]

        diff = obs[:, None, :] - mean
        outer = jnp.einsum('ijk, ijl -> ijkl', diff, diff)

        sigma = jnp.sum(gamma[..., None, None] * outer, axis=0)
        sigma = sigma / norm_const[:, None, None]

        return MultivariateGaussianModel(mean, sigma, self.is_log)

    def llhood(self, obs: Array) -> Array:
        return jax.vmap(multivariate_normal.pdf, in_axes=(None, 0, 0))(obs, self.mean, self.covariance)
    
    def logllhood(self, obs: Array) -> Array:
        return jax.vmap(multivariate_normal.logpdf, in_axes=(None, 0, 0))(obs, self.mean, self.covariance)
    
    def simulate(self, state: Array, uniform_sample: Array) -> Array:
        # TODO: Should this method be refactored? This cast into int seems
        # to be an unnecessary move, for questionable efficiency gains in the
        # discrete case.
        float_type = jnp.result_type(uniform_sample)
        int_type = jnp.int64 if float_type == jnp.float64 else jnp.int32
        seed = jax.lax.bitcast_convert_type(uniform_sample, int_type)
        return jax.random.multivariate_normal(jax.random.key(seed), self.mean[state], self.covariance[state])
    
    def check_obs_compatibility(self, obs: Array) -> bool:
        return jnp.issubdtype(obs.dtype, jnp.floating)

    def to_log(self) -> Self:
        return MultivariateGaussianModel(self.mean, self.covariance, is_log=True)

    def to_prob(self) -> Self:
        return MultivariateGaussianModel(self.mean, self.covariance, is_log=False)
    
    def astype(self, dtype) -> Self:
        return MultivariateGaussianModel(self.mean.astype(dtype), self.covariance.astype(dtype), is_log=self.is_log)
    
    @property
    def is_valid(self) -> bool:
        nan_parameters = (
            jnp.any(jnp.isnan(self.mean)) &
            jnp.any(jnp.isnan(self.covariance))
        )

        likelihoods = jax.vmap(
            multivariate_normal.pdf,
        )(jnp.zeros_like(self.mean), self.mean, self.covariance)

        nan_likelihoods = jnp.any(jnp.isnan(likelihoods))

        return (not nan_parameters) and (not nan_likelihoods)
    
    @property
    def has_multiple_outputs(self) -> bool:
        return True
        
    def __str__(self):
        return f'''Gaussian observation model(\n\tmean = \n{self.mean}\n\n\tcovariance = \n{self.covariance}\n)'''
    
    def construct_frozen_parameter_pytree(self, mask: Array) -> Self:
        '''Construct a pytree suitable for mapped masking of parameters.'''
        mean_mask = mask[..., 0]
        covariance_mask = mask[..., 1:]

        return MultivariateGaussianModel(mean_mask, covariance_mask, self.is_log)
    
    def get_params(self) -> Array:
        return jnp.concat([self.mean[..., None], self.covariance], axis=-1)
    
    def squeeze(self) -> Self:
        '''Does nothing for this model, but still present for compatibility'''
        return self

    ### TODO:
    
    @property
    def ndim(self) -> int:
        raise NotImplementedError
    
    @property
    def dtype(self) -> jax.typing.DTypeLike:
        raise NotImplementedError