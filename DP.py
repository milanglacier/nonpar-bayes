# %%

from msilib.schema import Component
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.stats as st  # type: ignore
from typing import List, Dict, Tuple
import numba
from numba import float64, int64


# %%
# %%

spec = [

    ("y", numba.typeof(np.array([1.0]))),
    ("sigma2", float64),
    ("m0", float64),
    ("D", float64),
    ("a", float64),
    ("b", float64),
    ("c", float64),
    ("d", float64),
    ("initial_components", int64),
    ("iters", int64)


]

@numba.experimental.jitclass(spec)
class DPGMM():
    """
    Dirichlet process gaussian mixture model.
    Currently the 1-dimensional case
    with observations having fixed prior variance is implemented.
    Assume a `G ~ DP(M, G_0), theta_i ~ G` prior, and `G_0 ~ N(m, B)`

    Params:

    y: np array of n x 1.

    sigma2: float, the prior of variance of `y_i`,
        assuming `y_i | theta_i ~ N(theta_i, sigma^2)`.

    m0: float, the prior mean of the mean of the base measure `G_0`: m.

    D: float, the prior variance of the mean of the base measure `G_0`: m.

    a: float, the gamma prior parameter of
        the variance of the base measure `G_0`: B.

    b: float, the gamma prior parameter of
        the variance of the base measure `G_0`: B.

    c: float, the gamma prior parameter of
        the scale parameter of the DP: M.

    d: float, the gamma prior parameter of
        the scale parameter of the DP: M.

    initial_components: int, the number of the components at the inital.

    iters: int, the size of posterior sample.

    """

    def __init__(self, y: np.array, sigma2: float,
                 m0: float, D: float, a: float, b: float,
                 c: float, d: float, initial_components: int,
                 iters: int) -> None:
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

        # set the hyper parameters need to be sampled
        # m: the mean of `G_0`, B: the variance of `G_0`
        # M: the scaled parameter, eta: the auxiliary variable
        # self.hyperparams_list = ["m", "M", "B", "eta"]

    def initialize(self):
        """
        initialize the value of the parameters to be sampled.

        return:
        hyperparms_value: the initial value of hyperparams
        s: the initial value of class membership
        theta: the initial value of class center
        """
        m = np.random.normal(loc=self.m0, scale=np.sqrt(self.D))
        M = np.random.gamma(shape=self.c, scale=1 / self.d)
        B = 1 / np.random.gamma(shape=self.a, scale=1 / self.b)
        eta = np.random.beta(M+1, len(self.y))
        hyperparams_value = np.array([m, M, B, eta])

        # initialize the components membership
        s = np.random.choice(self.initial_components, len(self.y))
        # since if we random sample n sample from 1:K
        # the sample components id might be
        # [0, 0, 1, 1, 1, 4, 4, 8]
        # so we want to rearrange
      # the sample components id to be
        # [0, 0, 1, 1, 1, 2, 2, 3]
        s = self.rearrange_s(s)
        theta_star = np.random.normal(
            loc=m, scale=np.sqrt(B), size=self.initial_components)

        # initialize the corresponding center for each data point
        theta = np.array([theta_star[i] for i in s])

        return hyperparams_value, s, theta
        
    def rearrange_s(self, s):
        s_unique = np.unique(s)
        for i in range(len(s_unique)):
            s[s == s_unique[i]] = i
        return s

    def sample(self) -> Tuple[np.array, np.array, np.array]:

        hyperparams_value, s, theta = self.initialize()
        hyperparams_sample = np.empty((self.iters, 4))
        s_sample = np.empty((self.iters, len(self.y)))
        theta_sample = np.empty((self.iters, len(self.y)))
        print(s)

        for i in range(self.iters): 
            s = self.update_s(s, 
                hyperparams_value[0], hyperparams_value[1], hyperparams_value[2])

        return s

    def update_s(self, s: np.array, m: float, M: float, B: float) -> np.array:

        for i in range(len(s)):
            s_minus = np.delete(s, i)
            y_minus = np.delete(self.y, i)
            
            components_minus = np.unique(s_minus)
            # the unique components id excluding y_i
            n_j_minus, v_j_minus, m_j_minus = \
            self.calculate_n_v_m_at_j(components_minus, s_minus, m, B, y_minus)

            weights_at_j = (n_j_minus *
                            self.dnorm(self.y[i], mean=m_j_minus,
                                        std=np.sqrt(v_j_minus + self.sigma2)))

            weights_at_new = M * self.dnorm(self.y[i],
                                            mean=m,
                                            std=np.sqrt(B + self.sigma2))
            weights = (np.append(weights_at_j, weights_at_new) /
                (weights_at_j.sum() + weights_at_new))
            
            s_i = self.sample_by_prob(
                np.append(components_minus, components_minus.max()+1),
                prob=weights)
            s[i] = s_i
        
        # after update s, rearrange the id
        s = self.rearrange_s(s) 
        return s 
    
    def update_theta(self, theta, s, m, M, B):
        components = np.unique(s)
        n_j, v_j, m_j = self.calculate_n_v_m_at_j(components, s, m, B, self.y)

        for i in range(len(theta)):
            pass
            
                                         
    def calculate_n_v_m_at_j(self, components, s, m, B, y):

        # the number of observations in each components
        n_j = np.array(
            [(s == i).sum()
                for i in components])

        sum_of_y_in_group_j = np.array(
            [y[s == i].sum()
                for i in components]
        )

        v_j = 1 / (1/B + n_j/self.sigma2)
        m_j = v_j * (m/B + sum_of_y_in_group_j/self.sigma2)
            
        return n_j, v_j, m_j
        
    def dnorm(self, x, mean, std):
        return np.exp(-((x-mean)/std)**2/2) / (std * np.sqrt(2 * np.pi))
        
    def sample_by_prob(self, x, prob):
        cumprob = np.cumsum(prob)
        unif = np.random.rand()
        y = x[-1]
        for i in range(len(cumprob)):
            if unif < cumprob[i]:
                y = x[i]
            break
        return y


# %%
y = np.append(np.random.normal(loc = 1, size = 15), np.random.normal(loc = 4, size = 15))
a = DPGMM(y = y,
          sigma2=1.0, m0=2.5, D=1.0, a=1.0, b=1.0, c=1.0, d=1.0,
          initial_components=2, iters=1000)
a.initialize()
a.sample()

# %%
