from .models import HiddenMarkovParameters, FreezeConfig, FreezeMasks, assert_valid_hmm, DiscreteModel, MultivariateGaussianModel
from .algorithms import baum_welch, forward_backward, generate_sequence, likelihood, log_likelihood
from ._precision import PrecisionWarning
from .util import normalize_rows