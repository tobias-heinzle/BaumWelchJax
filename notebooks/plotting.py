

from typing import Optional
from baum_welch_jax.models.hmm import HiddenMarkovParameters
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_inference_statistics(result):
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(12,3)
    ax[0].semilogy(np.diff(result.log_likelihoods))
    ax[0].set_title('Increase in log likelihood')
    ax[1].semilogy(np.abs(result.log_likelihoods[:result.iterations]))
    ax[1].set_title('Negative log likelihood')
    ax[2].semilogy(result.residuals)
    ax[2].set_title('Residuals $||\\theta_t - \\theta_{{t-1}}||_\\infty$')
    fig.tight_layout()
    return fig, ax

def plot_hmm_params(
        hmm: HiddenMarkovParameters, 
        plot_mu: bool=True, 
        with_numbers: bool=False, 
        cmap: str='viridis',
        ax: Optional[np.array] = None
        ) -> tuple[Figure | None, np.ndarray[Axes]]:
    n, m = hmm.O.shape

    cm = plt.colormaps[cmap]

    if hmm.is_log:
        hmm = hmm.to_prob()

    if plot_mu:
        if hmm.mu.ndim == 1:
            n_mu = 1
            _mu = hmm.mu[:,None]
        else:
            n_mu = hmm.mu.shape[0]
            _mu = hmm.mu.T

        width_ratios = (n, m, n_mu)
        
    else:
        width_ratios = (n, m)
    
    if ax is None:
        fig, ax = plt.subplots(1, len(width_ratios), width_ratios=width_ratios, constrained_layout=True)
        fig.set_size_inches((2.3, 2.3))
    else:
        fig = None

    ax[0].matshow(hmm.T, cmap=cmap)
    ax[0].set_title("T")

    ax[1].matshow(hmm.O, cmap=cmap)
    ax[1].set_title("O")


    if plot_mu:
        ax[2].matshow(_mu, cmap=cmap)
        ax[2].set_title(r"$\mu$")
    
    if with_numbers:
        for idx, mat in enumerate((hmm.T, hmm.O, hmm.mu)):
            if idx == 2 and not plot_mu:
                break
            if not mat.ndim >= 2:
                mat = mat[..., None]
            for (i, j), val in np.ndenumerate(mat):
                if val >= 1e-10:
                    fill_str = f"{val:.1g}" if val < 0.9 else f"{val:.5f}"
                    fill_str = fill_str.lstrip('0')
                    if val == 1.0:
                        fill_str = '1'
                    ax[idx].text(
                        j, i, fill_str, 
                        ha="center", va="center", 
                        fontsize=7,  # same as ticks
                        color=cm(0.0) if val > mat.max()/2 else cm(1.0)
                )

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    
    return fig, ax



def plot_stats(states, observations):
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(16,4)

    ax[0].set_title("Mean state evolution over time")
    ax[0].plot(np.mean(states, axis=0))
    ax[0].grid()
    ax[0].set_xlabel("Time step")
    ax[0].set_ylabel("State")

    
    ax[1].set_title("Arrival distribution in final state")
    ax[1].plot(np.mean(states == np.max(states), axis=0))
    ax[1].set_xlabel("Sequence length")
    ax[1].set_ylabel("Probability of reaching final state before $t$")
    ax[1].grid()

    obs_distribution = np.array(
        list(
            map(
                lambda o: np.count_nonzero(observations == o, axis=0), 
                np.arange(np.max(observations) + 1)
                )
            )
        ) / observations.shape[0]
    for k, obs_prob in enumerate(obs_distribution):
        ax[2].plot(obs_prob, label=f"$o^{k}$")
    ax[2].set_title("Observation distribution over time")
    ax[2].grid()
    ax[2].legend()
    ax[2].set_xlabel("Time step")
    ax[2].set_ylabel("$p(o_t)$")

    state_distribution = np.array(
        list(
            map(
                lambda s: np.count_nonzero(states == s, axis=0),
                np.arange(np.max(states) + 1)
                )
            )
        ) / states.shape[0]
    for k, state_prob in enumerate(state_distribution):
        ax[3].plot(state_prob, label=f"$s^{k}$")

    ax[3].set_title("State distribution over time")
    ax[3].set_xlabel("t")
    ax[3].set_ylabel("$p(s_t)$")
    ax[3].grid()
    ax[3].legend()

    plt.tight_layout()
    plt.show()

