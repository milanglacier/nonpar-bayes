# %%

import numpy as np  # type: ignore
from typing import Tuple
import pandas as pd  # type: ignore
import numba  # type: ignore
from numba import float64, int64

# %%
# %%
# spec = [("y", numba.typeof(np.array([1.0]))), ("sigma2", float64),
#         ("m0", float64), ("D", float64), ("a", float64), ("b", float64),
#         ("c", float64), ("d", float64), ("initial_components", int64),

spec = [("y", numba.typeof(np.array([[1.0, 1.0], [1.0, 1.0]]))),
        ("x", numba.typeof(np.array([[1.0, 1.0], [1.0, 1.0]]))),
        ("M", float64),
        ("mu", numba.typeof(np.array([[1.0, 1.0], [1.0, 1.0]]))),
        ("initial_components", int64),
        ("iters", int64)]


# @numba.experimental.jitclass(spec)
class ANOVADDP():
    """
    ANOVA DDP model

    Params:

    `y`: np array of `n * p`, p the dimension of response,
        n is the number of observation.

    `x`: np array of `n * q`, q the dimension of covariate,
        i.e the combination of treatments level.

    `M`: float, the mass parameter of the DP.

    `mu`: np array of `p * q`, p the dimension of response,
        q the dimension of the covariate,
        mu is the prior mean of the theta,
        i.e the mean of `p^0`

    `initial_components`: int, the number of the components at the initial.

    `iters`: int, the size of posterior sample.

    """

    def __init__(self, y: np.ndarray, x: np.ndarray,
                 mu: np.ndarray, M: float,
                 initial_components: int, iters: int) -> None:
        self.y = y
        self.x = x
        self.mu = mu
        self.M = M
        self.initial_components = initial_components
        self.iters = iters

    def initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        initialize the value of the parameters to be sampled.

        return:
        `s`: np.array of length `n`, the initial value of membership
        `theta`: np.array of `n * p * q`,
            the initial value of components center
        `theta_star`: np.array of `k * p * q`,
            the k unique values of theta out of n values.
        """

        # initialize the components membership
        s = np.random.choice(self.initial_components, self.y.shape[0])
        s = self.rearrange_s(s)

        # initialize the corresponding center for each data point
        theta_star = np.random.normal(loc=self.mu,
                                      size=[self.initial_components,
                                            self.mu.shape[0],
                                            self.mu.shape[1]
                                            ]
                                      )
        theta = np.zeros([self.y.shape[0], self.mu.shape[0], self.mu.shape[1]])

        for i in range(self.y.shape[0]):
            theta[i, :, :] = theta_star[s[i], :, :]

        return s, theta

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
        `s_sample`: np.array of shape `(iters, n)` related to the membership
            for each datapoints.
        `theta_sample`: np.array of shape `(iters, n, p, q)` related
            the parameters we are interested.
        `newy_sample`: np.array of shape `(iters, h, 2)`,
            h is the number of all the possible combination of treatments,
            i.e cartesian product of v = 1,2,...,V, w = 1,2,...W
            sample from the predictive sample of newx,
            where newx has h values.

        '''
        s, theta = self.initialize()
        s_sample = np.empty((self.iters, self.y.shape[0]), dtype=np.int64)
        theta_sample = np.empty((self.iters,
                                 self.y.shape[0],
                                 self.mu.shape[0],
                                 self.mu.shape[1]),
                                dtype=np.float64)

        newx = np.array([
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1]
        ])

        newy_sample = np.empty((self.iters,
                                newx.shape[0],
                                self.y.shape[1]), dtype=np.float64)

        for i in range(self.iters):

            s = self.update_s(s, theta)
            theta, theta_star, s_unique = self.update_theta(theta, s)

            newy = self.predictive_y(theta_star, s, s_unique, newx)

            s_sample[i, :] = s
            theta_sample[i, :, :, :] = theta
            newy_sample[i, :, :] = newy

        return s_sample, theta_sample, newy_sample

    def update_s(self, s: np.ndarray,
                 theta: np.ndarray) -> np.ndarray:
        """
        The gibbs sampler for updating s the membership array.
        """

        for i in range(s.shape[0]):
            s_minus = np.delete(s, i, axis=0)
            # y_minus = np.delete(self.y, i, axis=0)

            # n_j_minus is the number of observation in each group,
            # each group consists of the same value of s.
            s_unique_minus, n_j_minus = np.unique(s_minus, return_counts=True)

            # the unique value of theta
            theta_star_minus = np.array(
                [theta[self.index(s_minus, i), :, :] for i in s_unique_minus]
            )

            # the mean of y_i if y_i belongs to each group j
            mu_i_at_each_group = theta_star_minus @ self.x[i, :]

            # get the density of y[:, 0]

            yi0 = self.y[i, 0]
            mu_i0_at_each_group = mu_i_at_each_group[:, 0]
            density_i0_at_each_group = self.dnorm(yi0,
                                                  mean=mu_i0_at_each_group,
                                                  std=1)

            # get the density of y[:, 1]

            yi1 = self.y[i, 1]
            mu_i1_at_each_group = mu_i_at_each_group[:, 1]
            density_i1_at_each_group = self.dnorm(yi1,
                                                  mean=mu_i1_at_each_group,
                                                  std=1)
            density_at_each_group = n_j_minus * (density_i0_at_each_group
                                                 * density_i1_at_each_group)

            # the probability that s_i will form a new group

            mu_i_overall = self.mu @ self.x[i, :]
            density_yi_new_group = (self.M *
                                    self.dnorm(
                                        self.y[i, :], mean=mu_i_overall, std=4)
                                    .prod()
                                    )

            # bind the two group and normalize

            prob_belongs_to_j_or_new_group = np.append(
                density_at_each_group, density_yi_new_group)

            prob_belongs_to_j_or_new_group = (prob_belongs_to_j_or_new_group
                                              / prob_belongs_to_j_or_new_group.sum())

            groups_add_new = np.append(s_unique_minus,
                                       s_unique_minus.max() + 1)

            s_i = self.sample_by_prob(groups_add_new,
                                      prob_belongs_to_j_or_new_group)

            s[i] = s_i

        # after update s, rearrange the id
        s = self.rearrange_s(s)
        return s

    def index(self, array: np.ndarray, item: int) -> int:
        """
        given an array [1,2,5,5],
        index(array, 5) will return 2
        """
        for idx, val in np.ndenumerate(array):
            if val == item:
                return idx[0]
        return -1

    def update_theta(self,
                     theta: np.ndarray,
                     s: np.ndarray) -> Tuple[np.ndarray,
                                             np.ndarray,
                                             np.ndarray]:
        """
        The gibbs sampler for updating theta the components center array.

        returns:

        `theta`: np array of `n * p * q`, the posterior sample of theta
            at this round.
        `theta_star`: np array of `k * p * q`, the posterior sample of
            unique values of theta
        `s_unique`: np array of length `k`, the unique value
            of class membership
        """
        s_unique = np.unique(s)

        theta_star = np.empty((s_unique.shape[0],
                               self.mu.shape[0],
                               self.mu.shape[1]), dtype=np.float64)

        for s_idx in s_unique:

            y0_group_j = self.y[s == s_idx, 0]
            y1_group_j = self.y[s == s_idx, 1]
            x_group_j = self.x[s == s_idx, :]

            # the covariance matrix of alpha_0 (p * 1)
            # and the covariance matrix of alpha_0 (p * 1)
            # they are the same

            # prior covmat of theta
            Sigma_alpha = np.eye(self.mu.shape[1])
            inv = np.linalg.inv
            mu_0 = self.mu[0, :]
            mu_1 = self.mu[1, :]

            mean_theta_star_j_0 = (inv(inv(Sigma_alpha) + x_group_j.T @ x_group_j) @
                                   (inv(Sigma_alpha) @ mu_0 + x_group_j.T @ y0_group_j))

            mean_theta_star_j_1 = (inv(inv(Sigma_alpha) + x_group_j.T @ x_group_j) @
                                   (inv(Sigma_alpha) @ mu_1 + x_group_j.T @ y1_group_j))

            # posterior covmat of theta
            Sigma_alpha_after = inv(inv(Sigma_alpha) + x_group_j.T @ x_group_j)

            theta_star_j_0 = np.random.multivariate_normal(mean=mean_theta_star_j_0,
                                                           cov=Sigma_alpha_after)

            theta_star_j_1 = np.random.multivariate_normal(mean=mean_theta_star_j_1,
                                                           cov=Sigma_alpha_after)

            theta_star_j = np.array([theta_star_j_0, theta_star_j_1])
            theta_star[s_idx, :, :] = theta_star_j

        for i in range(self.y.shape[0]):
            theta[i, :, :] = theta_star[s[i], :, :]

        return theta, theta_star, s_unique

    def predictive_y(self, theta_star, s, s_unique, newx) -> np.array:
        """
        given a new x, return a sample from the predictive distribution,
        i.e P(y|all_params, newx)

        params:

        newx: np array of `h * q`, h the number of new observed x,
            q the dimension of covariates

        return:

        newy: np.array of `h * p`,
            there will be h newx, corresponding to v = 1,2, ...V
            and w = 1,2,...W
            and p is the dimension of y
        """

        prob_select_group_j = (
            np.array([(s == i).sum() for i in s_unique])
        )

        prob_added_new = np.append(prob_select_group_j, self.M)
        prob_added_new = prob_added_new / prob_added_new.sum()
        group_added_new = np.append(s_unique, s_unique.max() + 1)

        selected_group = self.sample_by_prob(group_added_new, prob_added_new)

        if selected_group == group_added_new.max():
            # generate a sample from thetas which comes frombase measure
            # sample y_0, the first dimension of y
            mean_newy_0 = newx @ self.mu[0, :]
            newy_0 = np.random.normal(loc=mean_newy_0, size=newx.shape[0])

            # sample y_1, the second dimension of y
            mean_newy_1 = newx @ self.mu[1, :]
            newy_1 = np.random.normal(loc=mean_newy_1, size=newx.shape[0])

            newy = np.array([newy_0, newy_1]).T
            # after transpose, the dimension becomes of h * 2

        else:
            theta_star_j = theta_star[selected_group, :]
            # sample y_0, the first dimension of y
            theta_star_j_0 = theta_star_j[0, :]
            mean_newy_0 = newx @ theta_star_j_0
            newy_0 = np.random.normal(loc=mean_newy_0, size=newx.shape[0])

            # sample y_1, the second dimension of y
            theta_star_j_1 = theta_star_j[1, :]
            mean_newy_1 = newx @ theta_star_j_1
            newy_1 = np.random.normal(loc=mean_newy_1, size=newx.shape[0])

            newy = np.array([newy_0, newy_1]).T
            # after transpose, the dimension becomes of h * 2

        return newy

    def dnorm(self, x: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
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
# %%

if __name__ == "__main__":

    y = pd.read_csv("mydata.csv")
    y = np.array(y).T
    x = pd.read_csv("d.csv")
    x = np.array(x).T
    mu = np.array([
        [3, 1.25, 8.5/2, -3.5/2, -3.5/2],
        [1, 13.5/2, 13.5/2, 15.5/2, 9.5/2]
    ])

    anovaddp = ANOVADDP(y=y, x=x, M=0.001, iters=1000,
                        mu=mu, initial_components=2)
    s, theta = anovaddp.initialize()
    s = anovaddp.update_s(s, theta)
    theta = anovaddp.update_theta(theta, s)

    s, theta, newy = anovaddp.sample()

    y_v1w1 = newy[:, 0, :]
    pd.DataFrame(y_v1w1).to_csv("y_v1w1.csv", index=False)

    y_v2w1 = newy[:, 1, :]
    pd.DataFrame(y_v2w1).to_csv("y_v2w1.csv", index=False)

    y_v1w2 = newy[:, 2, :]
    pd.DataFrame(y_v1w2).to_csv("y_v1w2.csv", index=False)

    y_v2w2 = newy[:, 3, :]
    pd.DataFrame(y_v1w1).to_csv("y_v2w2.csv", index=False)

    # %%
