import jax.numpy as jnp
from jax.lax import cond, scan


def joint_likelihood(distributions, obs, k):
    """
    Computes the joint liklihood of a sequence of observations given the distributions T, O and mu
    """
    T, O, mu = (distributions["T"], distributions["O"], distributions["mu"])
    initial_carry = {
        "carry_factor": mu.T,
        "likelihood": 1.0
    }

    def loop_body(carry, obs):

        factor = carry["current_state_distribution"]
        llhood = carry["liklihood"]

        # factor, llhood = carry

        # def update(obs):

        state_distribution = factor @ T
        obs_distr = state_distribution @ O

        return (
            {
                "carry_factor": state_distribution * O[:, obs],
                "likelihood": llhood * obs_distr[obs] / jnp.sum(obs_distr)
            },

        )

        # factor, llhood = update(obs)

        # # factor, llhood = cond(
        # #     j <= k,
        # #     update,
        # #     lambda *_: (factor, llhood),
        # #     obs
        # # )

        # return ((factor, llhood), llhood)

    carry, _ = scan(
        loop_body,
        initial_carry,
        obs,
        unroll=False
    )

    _, _, llhood = carry

    return llhood


def joint_likelihood_numpy(distributions, obs, actions, k):
    T, O, mu = (distributions["T"], distributions["O"], distributions["mu"])

    def loop_body(carry, data):
        j, factor, llhood = carry
        action, obs = data

        def update(args):
            a, obs = args

            new_factor = factor @ T[:, a, :]
            obs_distr = new_factor @ O

            return (
                new_factor * O[:, obs],
                llhood * obs_distr[obs] / jnp.sum(obs_distr)
            )

        if j <= k:
            factor, llhood = update((action, obs))

        return ((j + 1, factor, llhood), action)

    i = 0
    _mu = mu.T
    llhood = 1.0

    for data in zip(actions, obs):
        (i, _mu, llhood), _ = loop_body((i, _mu, llhood), data)

    return llhood


def single_likelihood(distributions, obs, actions, k):
    T, O, mu = (distributions["T"], distributions["O"], distributions["mu"])

    def loop_body(carry, data):
        j, factor = carry
        action, obs = data

        def update(args):
            a, obs = args
            return factor @ T[:, a, :] * O[:, obs],

        factor = cond(
            j <= k - 1,
            update,
            lambda *_: factor,
            (action, obs)
        )

        return ((j + 1, factor), action)

    carry, _ = scan(
        loop_body,
        (0, mu.T),
        jnp.stack([actions, obs], axis=-1),
        unroll=False
    )

    _, factor = carry

    obs_distr = factor @ T[:, actions[k], :] @ O

    return obs_distr[obs[k]] / jnp.sum(obs_distr)


def conditional_likelihood(distributions, obs, action, state):
    T, O = (distributions["T"], distributions["O"])

    return T[state, action, :] @ O[:, obs]
