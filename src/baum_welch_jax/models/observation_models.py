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
        pass

    @abstractmethod
    def llhood(self, obs: Array) -> Array:
        '''Unnormalized likelihood of the states, given the observation.'''
        pass

    @abstractmethod
    def logllihood(self, obs: Array) -> Array:
        '''Unnormalized log likelihood of the states, given the observation.'''
        pass

    @abstractmethod
    def simulate(self, state: Array, uniform_sample: Array) -> Array:
        '''Transform a sample of a uniform distribution into a sample of the observation
        distribution.'''
        pass




@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscreteObservationModel(ObservationModel):
    obs_probs: Array
    is_log: bool

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