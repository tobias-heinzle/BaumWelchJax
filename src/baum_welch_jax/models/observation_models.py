from dataclasses import dataclass, field
from typing import NamedTuple, Any, Self
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.special import logsumexp
import jax


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
    def logllihood(self, obs: Array) -> Array:
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




@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscreteObservationModel(ObservationModel):
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

        return DiscreteObservationModel(O, self.is_log)


    def llhood(self, obs: Array) -> Array:
        if self.is_log:
            return jnp.exp(self.obs_probs[:, obs])
        
        return self.obs_probs[:, obs]
    
    def logllihood(self, obs: Array) -> Array:
        if self.is_log:
            return self.obs_probs[:, obs]

        return jnp.log(self.obs_probs[:, obs])
    
    def obs_cdf(self, state):
        # TODO: should this be field of the class, instead of computation 
        # for every sample?
        if self.is_log:
            return jnp.cumsum(jnp.exp(self.obs_probs[state]))
        
        return jnp.cumsum(self.obs_probs[state])
    
    def simulate(self, state: Array, uniform_sample: Array) -> Array:
        return jnp.argmax(self.obs_cdf(state) >= uniform_sample)
    
    def to_log(self):
        if self.is_log:
            raise ValueError('Attempted log conversion; Parameters are already logprobabilities.')
        return DiscreteObservationModel(jnp.log(self.obs_probs), True)
    
    def to_prob(self):
        if self.is_log:
            return DiscreteObservationModel(jnp.exp(self.obs_probs), False)
        raise ValueError('Attempted probability conversion; Parameters are already probabilities.')
    
    def astype(self, dtype):
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError("dtype must be floating point number")
        
        return DiscreteObservationModel(
            self.obs_probs.astype(dtype),
            self.is_log
        )
    
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
            all_sum_to_one = jnp.allclose(logsumexp(self.obs_probs, axis=1), 0.0)

            return jnp.all(jnp.array([correct_dims, all_sum_to_one]))

        else:
            all_positive = jnp.all(self.obs_probs >= 0)
            all_sum_to_one = jnp.allclose(jnp.sum(self.obs_probs, axis=1), 1.0)
              
            return jnp.all(jnp.array([correct_dims, all_positive, all_sum_to_one, is_float]))
    
    def __str__(self):
        return f'''Discrete observation model(\n\tobs_probs = \n\t{self.obs_probs}\n\n\ts_log = {self.is_log}\n)'''