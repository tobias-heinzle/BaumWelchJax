from typing import NamedTuple, Self

from jax import Array

from baum_welch_jax.models import HiddenMarkovParameters


class IterationState(NamedTuple):
    '''
    Structured tuple for the (intermediate) results of expectation maximization
    '''
    params: HiddenMarkovParameters
    log_likelihoods: Array
    residuals: Array
    iterations: int
    terminated: bool

    def squeeze(self) -> Self:
        return IterationState(
            params=HiddenMarkovParameters(
                self.params.T.squeeze(),
                self.params.O.squeeze(),
                self.params.mu.squeeze(),
                self.params.is_log
            ),
            log_likelihoods=self.log_likelihoods.squeeze(),
            residuals=self.residuals.squeeze(),
            iterations=self.iterations,
            terminated=self.terminated
        )
    
    def replace_mu(self, new_mu: Array) -> Self:
        return IterationState(
            params=self.params.replace_mu(new_mu),
            log_likelihoods=self.log_likelihoods,
            residuals=self.residuals,
            iterations=self.iterations,
            terminated=self.terminated
        )