# %%

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Tuple
import numba  # type: ignore
from numba import float64, int64
import matplotlib.pyplot as plt  # type: ignore

# %%
# %%
spec = [("y", numba.typeof(np.array([1.0]))), ("sigma2", float64),
        ("m0", float64), ("D", float64), ("a", float64), ("b", float64),
        ("c", float64), ("d", float64), ("initial_components", int64),
        ("iters", int64)]


@numba.experimental.jitclass(spec)
class DPGMM():
    """
    Dirichlet process gaussian mixture model.
    Currently the 1-dimensional case
    with observations having fixed prior variance is implemented.
    Assume a `G ~ DP(M, G_0), theta_i ~ G` prior, and `G_0 ~ N(m, B)`

    Params:

    `y`: np array of length n.

    `sigma2`: float, the prior of variance of `y_i`,
        assuming `y_i | theta_i ~ N(theta_i, sigma^2)`.

    `m0`: float, the prior mean of the mean of the base measure `G_0`: m.

    `D`: float, the prior variance of the mean of the base measure `G_0`: m.

    `a`: float, the gamma prior parameter of
        the variance of the base measure `G_0`: B.

    `b`: float, the gamma prior parameter of
        the variance of the base measure `G_0`: B.

    `c`: float, the gamma prior parameter of
        the scale parameter of the DP: M.

    `d`: float, the gamma prior parameter of
        the scale parameter of the DP: M.

    `initial_components`: int, the number of the components at the inital.

    `iters`: int, the size of posterior sample.

    """

    def __init__(self, y: np.ndarray, sigma2: float, m0: float, D: float,
                 a: float, b: float, c: float, d: float,
                 initial_components: int, iters: int) -> None:
        self.y = y
        # self.label = label,
        self.sigma2 = sigma2
        self.m0 = m0
        self.D = D
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.initial_components = initial_components
        self.iters = iters

    def initialize(self):
        """
        initialize the value of the parameters to be sampled.

        return:
        `hyperparms_value`: `[m, M, B, eta]`, the initial value of hyperparams
        `s`: np.array of length `n`, the initial value of membership
        `theta`: np.array of length `n`, np.array,
            the initial value of components center
        """
        m = np.random.normal(loc=self.m0, scale=np.sqrt(self.D))
        M = np.random.gamma(shape=self.c, scale=1 / self.d)
        B = 1 / np.random.gamma(shape=self.a, scale=1 / self.b)
        eta = np.random.beta(M + 1, len(self.y))
        hyperparams_value = np.array([m, M, B, eta])

        # initialize the components membership
        s = np.random.choice(self.initial_components, len(self.y))
        s = self.rearrange_s(s)
        theta_star = np.random.normal(loc=m,
                                      scale=np.sqrt(B),
                                      size=self.initial_components)

        # initialize the corresponding center for each data point
        theta = np.array([theta_star[i] for i in s])

        return hyperparams_value, s, theta

    def rearrange_s(self, s: np.ndarray) -> np.array:
        """
        given a membership array s,
        which may take form like [0, 0, 1, 1, 1, 4, 4, 8]

        rearrange the membership array to be
        [0, 0, 1, 1, 1, 2, 2, 3]

        params:
        `s`: np.array, the membership array

        returns:
        `s`: np.array, the rearranged array.
        """
        s_unique = np.unique(s)
        for i in range(len(s_unique)):
            s[s == s_unique[i]] = i
        return s

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Implement the Gibbs sampler to obtain the posterior sample.

        return:
        `hyperparams_sample`: np.array of shape `(iters, 4)` related to
            `[m, M, B, eta]`.
        `s_sample`: np.array of shape `(iters, n)` related to the membership
            for each datapoints.
        `theta_sample`: np.array of shape `(iters, n)` related to components
            center for each datapoints.
        '''
        hyperparams_value, s, theta = self.initialize()
        # hyperparams_value = [12, 0.1, 0.5, 1]
        hyperparams_sample = np.empty((self.iters, 4), dtype=np.float64)
        s_sample = np.empty((self.iters, len(self.y)), dtype=np.int64)
        theta_sample = np.empty((self.iters, len(self.y)), dtype=np.float64)

        for i in range(self.iters):
            s = self.update_s(
                s,
                hyperparams_value[0],
                hyperparams_value[1],
                hyperparams_value[2])

            theta = self.update_theta(
                theta,
                s,
                hyperparams_value[0],
                hyperparams_value[2])

            hyperparams_value = self.update_hyperparams(
                theta,
                hyperparams_value[0], hyperparams_value[1],
                hyperparams_value[2], hyperparams_value[3]
            )

            s_sample[i, :] = s
            theta_sample[i, :] = theta
            hyperparams_sample[i, :] = hyperparams_value

        return hyperparams_sample, s_sample, theta_sample

    def update_s(self, s: np.ndarray, m: float, M: float,
                 B: float) -> np.ndarray:
        """
        The gibbs sampler for updating s the membership array.
        """

        for i in range(len(s)):
            s_minus = np.delete(s, i)
            y_minus = np.delete(self.y, i)

            # the unique components id excluding y_i
            components_minus = np.unique(s_minus)

            n_j_minus, v_j_minus, m_j_minus = \
                self.calculate_n_v_m_at_j(
                    components_minus, s_minus, m, B, y_minus)

            weights_at_j = (n_j_minus *
                            self.dnorm(self.y[i],
                                       mean=m_j_minus,
                                       std=np.sqrt(v_j_minus + self.sigma2)))

            weights_at_new = M * self.dnorm(
                self.y[i], mean=m, std=np.sqrt(B + self.sigma2))
            weights = (np.append(weights_at_j, weights_at_new) /
                       (weights_at_j.sum() + weights_at_new))

            s_i = self.sample_by_prob(np.append(components_minus,
                                                components_minus.max() + 1),
                                      prob=weights)
            s[i] = s_i

        # after update s, rearrange the id
        s = self.rearrange_s(s)
        return s

    def update_theta(self, theta: np.ndarray, s: np.ndarray, m: float,
                     B: float) -> np.ndarray:
        """
        The gibbs sampler for updating theta the components center array.
        """
        components = np.unique(s)
        theta_star = np.zeros_like(components, dtype=np.float64)

        for i in range(len(components)):
            sum_y_j = self.y[s == components[i]].sum()
            n_j = (s == components[i]).sum()
            theta_mean = (self.sigma2 * m + B * sum_y_j) / (self.sigma2 +
                                                            B * n_j)

            theta_var = (self.sigma2 * B) / (self.sigma2 + B * n_j)
            theta_star[i] = np.random.normal(loc=theta_mean,
                                             scale=np.sqrt(theta_var))
            theta[s == components[i]] = theta_star[i]

        return theta

    def update_hyperparams(self, theta: np.ndarray, m: float, M: float,
                           B: float, eta: float) -> np.ndarray:

        theta_star = np.unique(theta)
        k = len(theta_star)

        # update m
        D1 = 1 / (1 / self.D + k / B)
        m1 = D1 * (self.m0 / self.D + theta_star.sum() / B)
        m = np.random.normal(m1, np.sqrt(D1))

        # update B
        a1 = self.a + k / 2
        b1 = self.b + ((theta_star - m)**2).sum() / 2
        B = 1 / np.random.gamma(shape=a1, scale=1 / b1)

        # update eta
        eta = np.random.beta(M + 1, self.y.shape[0])

        # update M
        weight = ((self.c + k - 1) / (self.c + k - 1 + self.y.shape[0] *
                                      (self.d - np.log(eta))))
        unif = np.random.rand()
        if unif < weight:
            c1 = self.c + k
            d1 = self.d - np.log(eta)
            M = np.random.gamma(shape=c1, scale=1 / d1)

        else:
            c1 = self.c + k - 1
            d1 = self.d - np.log(eta)
            M = np.random.gamma(shape=c1, scale=1 / d1)

        return np.array([m, M, B, eta])

    def calculate_n_v_m_at_j(self, components: np.ndarray, s: np.ndarray,
                             m: np.ndarray, B: np.ndarray, y: np.ndarray):
        """
        calculate some important statistics for updating s.
        especially the mean(`m_j`) and variance(`v_j`)
            of center for each components
        (excluding the ith data point).
        """

        # the number of observations in each components
        n_j = np.array([(s == i).sum() for i in components])

        sum_of_y_in_group_j = np.array([y[s == i].sum() for i in components])

        v_j = 1 / (1 / B + n_j / self.sigma2)
        m_j = v_j * (m / B + sum_of_y_in_group_j / self.sigma2)

        return n_j, v_j, m_j

    def dnorm(self, x: float, mean: np.ndarray, std: np.ndarray) -> float:
        """
        since numba.jitclass does not support scipy.stat
        a simple function to calculate the normal density
        """
        return np.exp(-((x - mean) / std)**2 / 2) / (std * np.sqrt(2 * np.pi))

    def sample_by_prob(self, x: np.ndarray, prob: np.ndarray) -> int:
        """
        since numba.jitclass does not support np.random.choice
            with argument `p`.
        a simple function to implement sample with probability.
        """
        cumprob = np.cumsum(prob)
        unif = np.random.rand()
        y = x[-1]
        for i in range(len(cumprob)):
            if unif < cumprob[i]:
                y = x[i]
                break
        return y


# %%
# %%

y = np.append(np.random.normal(loc=16, size=50),
              np.random.normal(loc=9, size=50))
a = DPGMM(y=y,
          sigma2=1.0,
          m0=2.5,
          D=1.0,
          a=1.0,
          b=1.0,
          c=1.0,
          d=1.0,
          initial_components=2,
          iters=5000)
a.initialize()
hyperparams, s, theta = a.sample()
# %%
