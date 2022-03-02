## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = FALSE, warning = F, message = F)
options(scipen = 10)


## ---- eval = F----------------------------------------------------------------
## library(tidyverse)
## library(reticulate)
## use_python("/opt/homebrew/Caskroom/miniforge/base/bin/python")


## -----------------------------------------------------------------------------
library(tidyverse)
load("post.starting.RData")
y = as.numeric(y)
hypers <- posterior_sample[[1]][seq(2000, 10000, by = 5), ]
comps <- posterior_sample[[2]][seq(2000, 10000, by = 5), ]
theta <- posterior_sample[[3]][seq(2000, 10000, by = 5), ]


## ---- eval = F----------------------------------------------------------------
## dp <- import("DP")
## np <- import("numpy")
## DPGMM <- dp$DPGMM


## ---- eval = F----------------------------------------------------------------
## y <- np$array(c(rnorm(50, 5, 1), rnorm(50, -4, 1), rnorm(50, 13, 1)))
## id <- factor(c(rep(0, 50), rep(1, 50), rep(2, 50)))
## sigma2 <- 1.0
## m0 <- 1.0
## D <- 1.0
## a <- 1.0
## b <- 1.0
## c <- 1.0
## d <- 1.0
## initial_components <- 4
## iters <- 10000
## 
## 
## dpgmm <- DPGMM(
##     y, sigma2, m0, D,
##     a, b, c, d, initial_components, iters
## )
## 
## posterior_sample <- dpgmm$sample()
## save(y, id, posterior_sample, file = "post.starting.RData")


## ----startingexampley, fig.align = 'center', fig.height=3, fig.cap = 'The jittered plot of the synthetic data with color its corresponding membership', out.width='1\\linewidth'----
ggplot() +
    geom_jitter(aes(x = "y", y = y, color = id)) +
    theme(axis.title.x = element_blank()) +
    labs(x = "data points", color = "the actual membership")



## ---- include = FALSE---------------------------------------------------------

library(ggfortify)

theta1 = theta[, 1]
m = hypers[, 1]
theta2 = theta[, 2]
theta150 = theta[, 150]

acf1 = acf(theta1)
acfm = acf(m)



## ----startingexampleConv, fig.align = 'center', fig.height=4, fig.cap = 'The ACF and 2d density plot of the MC chain', out.width='1\\linewidth'----


cowplot::plot_grid(
    autoplot(acf1) + labs(x = "theta1"), 
    autoplot(acfm) + labs(x = "m"),
    ggplot() +
        stat_density_2d(aes(x = theta1, y = theta150,
                            fill = ..level..),
                        geom = "polygon") +
        scale_fill_gradient2(
            low = "#FFFAFA", mid = "#BBFFFF", high = "#483D8B"),

    ggplot() +
        stat_density_2d(aes(x = theta1, y = theta2,
                            fill = ..level..),
                        geom = "polygon") +
        scale_fill_gradient2(
            low = "#FFFAFA", mid = "#BBFFFF", high = "#483D8B"),
    ncol = 2
)



## ----startingexampleAvgtheta, fig.align='center', fig.height=4,fig.cap='Left: observed value vs observed value, Right: mean of theta vs observed value', out.width='1\\linewidth'----

theta_mean <- apply(theta, 2, mean)

cowplot::plot_grid(
ggplot() +
    geom_point(aes(x = y, y = y, color = id)) +
    labs(x = "the observed value",
         y = "the observed value"),

ggplot() +
    geom_point(aes(x = y, y = theta_mean, color = id)) +
    labs(x = "mean of theta for each data point",
         y = "the observed value")
)


## ----startingexampleAround, fig.align = 'center', fig.cap = 'The sampled membership and actuall membership in round 8000', out.width='1\\linewidth'----

ggplot() +
    geom_point(aes(
        x = 1:length(y), y = theta[1601, ],
        shape = id, color = factor(comps[1601, ])
    ), size = 1) +
    labs(shape = "actual component id", 
         color = "sampled component id",
         x = "index", y = "mean of theta")


## ----startingexampleNoComp, fig.align = 'center', fig.height=3, fig.cap = 'The barplot of the number of sampled membership', out.width='1\\linewidth'----
num_of_clusters <- apply(
    comps, 1,
    function(x) length(unique(x))
)

ggplot() +
    geom_bar(aes(x = num_of_clusters)) +
    scale_x_continuous(breaks = seq(1, 10, by = 1)) +
    labs(x = "number of components", y = "count")


## ---- startingexampleWCV------------------------------------------------------

freq_k = tibble(k = num_of_clusters) |>
    group_by(k) |>
    summarise(freq = n())

get_WCV = function(k, num_of_clusters) {

    iteration_with_K_eq_k = which(num_of_clusters == k)

    wcv_at_each_row = apply(theta[iteration_with_K_eq_k, ], 1,
        function(x) sum((x - y)^2)
    )


    sum(wcv_at_each_row)
}

freq_k = freq_k |>
    mutate(avg_wcv = sapply(k, 
            function(x) get_WCV(k = x,
                                num_of_clusters = num_of_clusters)) / freq)

get_true_wcv = function(y, true_center, true_membership) {

    wcv_at_each_component = sapply(true_center,
        function(x) sum((y[true_membership == x] - x)^2)
    )

    sum(wcv_at_each_component)
}


freq_k = freq_k |>
    mutate(true_wcv = get_true_wcv(
            y,
            c(5, -4, 13), 
            c(rep(5, 50), rep(-4, 50), rep(13, 50))
            ),
        reduced_proportion = 1 - avg_wcv / true_wcv
    )
knitr::kable(freq_k,
    caption = "The average WCV for different number of clusters among all iterations, and the true WCV (using the true center for each data point), and the the proportion of averaged WCV for each k to the true WCV")


## ---- eval = F----------------------------------------------------------------
## centers_1 = c(0, -5, 10, 5)
## centers_2 = seq(-25, 25, length = 10)
## centers_3 = centers_2
## 
## num_of_clusters_1 = c(50, 50, 50, 10)
## num_of_clusters_2 = rep(20, 10)
## num_of_clusters_3 = rep(100, 10)
## 
## membership_1 = purrr::map2(centers_1, num_of_clusters_1, ~rep(.x, .y)) |>
##     unlist()
## membership_2 = purrr::map2(centers_2, num_of_clusters_2, ~rep(.x, .y)) |>
##     unlist()
## membership_3 = purrr::map2(centers_3, num_of_clusters_3, ~rep(.x, .y)) |>
##     unlist()
## 
## y_1 = np$array(rnorm(160, mean = membership_1))
## y_2 = np$array(rnorm(200, mean = membership_2))
## y_3 = np$array(rnorm(1000, mean = membership_3))
## 
## 
## sigma2 <- 1.0
## m0 <- 1.0
## D <- 1.0
## a <- 1.0
## b <- 1.0
## c <- 1.0
## d <- 1.0
## initial_components <- 4
## iters <- 10000
## 
## 
## dpgmm_1 <- DPGMM(
##     y_1, sigma2, m0, D,
##     a, b, c, d, initial_components, iters
## )
## 
## posterior_sample_1 <- dpgmm_1$sample()
## 
## dpgmm_2 <- DPGMM(
##     y_2, sigma2, m0, D,
##     a, b, c, d, initial_components, iters
## )
## 
## posterior_sample_2 <- dpgmm_2$sample()
## 
## dpgmm_3 <- DPGMM(
##     y_3, sigma2, m0, D,
##     a, b, c, d, initial_components, iters
## )
## 
## posterior_sample_3 <- dpgmm_3$sample()
## 


## ---- eval = F----------------------------------------------------------------
## save(y_1, y_2, y_3,
##     posterior_sample_1, posterior_sample_2, posterior_sample_3,
##     file = "post.sample.3.RData")
## 


## -----------------------------------------------------------------------------
load("post.sample.3.RData")

centers_1 = c(0, -5, 10, 5)
centers_2 = seq(-25, 25, length = 10)
centers_3 = centers_2

num_of_clusters_1 = c(50, 50, 50, 10)
num_of_clusters_2 = rep(20, 10)
num_of_clusters_3 = rep(100, 10)

membership_1 = purrr::map2(centers_1, num_of_clusters_1, ~rep(.x, .y)) |>
    unlist()
membership_2 = purrr::map2(centers_2, num_of_clusters_2, ~rep(.x, .y)) |>
    unlist()
membership_3 = purrr::map2(centers_3, num_of_clusters_3, ~rep(.x, .y)) |>
    unlist()

y_1 = as.numeric(y_1)
y_2 = as.numeric(y_2)
y_3 = as.numeric(y_3)

hypers_1 <- posterior_sample_1[[1]][seq(2000, 10000, by = 5), ]
comps_1 <- posterior_sample_1[[2]][seq(2000, 10000, by = 5), ]
theta_1 <- posterior_sample_1[[3]][seq(2000, 10000, by = 5), ]


hypers_2 <- posterior_sample_2[[1]][seq(2000, 10000, by = 5), ]
comps_2 <- posterior_sample_2[[2]][seq(2000, 10000, by = 5), ]
theta_2 <- posterior_sample_2[[3]][seq(2000, 10000, by = 5), ]

hypers_3 <- posterior_sample_3[[1]][seq(2000, 10000, by = 5), ]
comps_3 <- posterior_sample_3[[2]][seq(2000, 10000, by = 5), ]
theta_3 <- posterior_sample_3[[3]][seq(2000, 10000, by = 5), ]


## -----------------------------------------------------------------------------

theta_mean_1 = apply(theta_1, 2, mean)
theta_mean_2 = apply(theta_2, 2, mean)
theta_mean_3 = apply(theta_3, 2, mean)



## ----ex2Avgtheta, fig.align='center', fig.height=8,fig.cap='Left: observed value vs observed value, Right: mean of theta vs observed value', out.width='1\\linewidth'----


y_vs_y = function(y, membership, scenerio) {


    ggplot() +
        geom_point(aes(x = y, y = y, 
                color = factor(as.integer(membership)))) +
        labs(x = paste0("observed value in scenerio ", scenerio),
             y = paste0("observed value in scenerio ", scenerio),
             color = "true center") +
        theme(axis.title = element_text(size = 8),
              legend.text = element_text(size = 8),
              legend.title = element_text(size = 8))

}

y_vs_theta = function(y, theta_mean, membership, scenerio) {


    ggplot() +
        geom_point(aes(x = y, y = theta_mean, 
                color = factor(as.integer(membership)))) +
        labs(x = paste0("mean of theta for each point in scenerio ", scenerio),
             y = paste0("observed value in scenerio ", scenerio),
             color = "") +
        theme(axis.title = element_text(size = 8),
              legend.text = element_text(size = 8),
              legend.title = element_text(size = 8))


}

cowplot::plot_grid(

    y_vs_y(y_1, membership_1, 1),
    y_vs_theta(y_1, theta_mean_1, membership_1, 1),
    y_vs_y(y_2, membership_2, 2),
    y_vs_theta(y_2, theta_mean_2, membership_2, 2),
    y_vs_y(y_3, membership_3, 3),
    y_vs_theta(y_3, theta_mean_3, membership_3, 3),
    nrow = 3
)


## # %%

## 
## import numpy as np  # type: ignore

## import pandas as pd  # type: ignore

## from typing import Tuple

## import numba  # type: ignore

## from numba import float64, int64

## import matplotlib.pyplot as plt  # type: ignore

## 
## # %%

## # %%

## spec = [("y", numba.typeof(np.array([1.0]))), ("sigma2", float64),

##         ("m0", float64), ("D", float64), ("a", float64), ("b", float64),

##         ("c", float64), ("d", float64), ("initial_components", int64),

##         ("iters", int64)]

## 
## 
## @numba.experimental.jitclass(spec)

## class DPGMM():

##     """

##     Dirichlet process gaussian mixture model.

##     Currently the 1-dimensional case

##     with observations having fixed prior variance is implemented.

##     Assume a `G ~ DP(M, G_0), theta_i ~ G` prior, and `G_0 ~ N(m, B)`

## 
##     Params:

## 
##     `y`: np array of length n.

## 
##     `sigma2`: float, the prior of variance of `y_i`,

##         assuming `y_i | theta_i ~ N(theta_i, sigma^2)`.

## 
##     `m0`: float, the prior mean of the mean of the base measure `G_0`: m.

## 
##     `D`: float, the prior variance of the mean of the base measure `G_0`: m.

## 
##     `a`: float, the gamma prior parameter of

##         the variance of the base measure `G_0`: B.

## 
##     `b`: float, the gamma prior parameter of

##         the variance of the base measure `G_0`: B.

## 
##     `c`: float, the gamma prior parameter of

##         the scale parameter of the DP: M.

## 
##     `d`: float, the gamma prior parameter of

##         the scale parameter of the DP: M.

## 
##     `initial_components`: int, the number of the components at the inital.

## 
##     `iters`: int, the size of posterior sample.

## 
##     """

## 
##     def __init__(self, y: np.ndarray, sigma2: float, m0: float, D: float,

##                  a: float, b: float, c: float, d: float,

##                  initial_components: int, iters: int) -> None:

##         self.y = y

##         # self.label = label,

##         self.sigma2 = sigma2

##         self.m0 = m0

##         self.D = D

##         self.a = a

##         self.b = b

##         self.c = c

##         self.d = d

##         self.initial_components = initial_components

##         self.iters = iters

## 
##     def initialize(self):

##         """

##         initialize the value of the parameters to be sampled.

## 
##         return:

##         `hyperparms_value`: `[m, M, B, eta]`, the initial value of hyperparams

##         `s`: np.array of length `n`, the initial value of membership

##         `theta`: np.array of length `n`, the initial value of components center

##         """

##         m = np.random.normal(loc=self.m0, scale=np.sqrt(self.D))

##         M = np.random.gamma(shape=self.c, scale=1 / self.d)

##         B = 1 / np.random.gamma(shape=self.a, scale=1 / self.b)

##         eta = np.random.beta(M + 1, len(self.y))

##         hyperparams_value = np.array([m, M, B, eta])

## 
##         # initialize the components membership

##         s = np.random.choice(self.initial_components, len(self.y))

##         s = self.rearrange_s(s)

##         theta_star = np.random.normal(loc=m,

##                                       scale=np.sqrt(B),

##                                       size=self.initial_components)

## 
##         # initialize the corresponding center for each data point

##         theta = np.array([theta_star[i] for i in s])

## 
##         return hyperparams_value, s, theta

## 
##     def rearrange_s(self, s: np.ndarray) -> np.array:

##         """

##         given a membership array s,

##         which may take form like [0, 0, 1, 1, 1, 4, 4, 8]

## 
##         rearrange the membership array to be

##         [0, 0, 1, 1, 1, 2, 2, 3]

## 
##         params:

##         `s`: np.array, the membership array

## 
##         returns:

##         `s`: np.array, the rearranged array.

##         """

##         s_unique = np.unique(s)

##         for i in range(len(s_unique)):

##             s[s == s_unique[i]] = i

##         return s

## 
##     def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

##         '''

##         Implement the Gibbs sampler to obtain the posterior sample.

## 
##         return:

##         `hyperparams_sample`: np.array of shape `(iters, 4)` related to

##             `[m, M, B, eta]`.

##         `s_sample`: np.array of shape `(iters, n)` related to the membership

##             for each datapoints.

##         `theta_sample`: np.array of shape `(iters, n)` related to components

##             center for each datapoints.

##         '''

##         hyperparams_value, s, theta = self.initialize()

##         # hyperparams_value = [12, 0.1, 0.5, 1]

##         hyperparams_sample = np.empty((self.iters, 4), dtype=np.float64)

##         s_sample = np.empty((self.iters, len(self.y)), dtype=np.int64)

##         theta_sample = np.empty((self.iters, len(self.y)), dtype=np.float64)

## 
##         for i in range(self.iters):

##             s = self.update_s(

##                 s,

##                 hyperparams_value[0],

##                 hyperparams_value[1],

##                 hyperparams_value[2])

## 
##             theta = self.update_theta(

##                 theta,

##                 s,

##                 hyperparams_value[0],

##                 hyperparams_value[2])

## 
##             hyperparams_value = self.update_hyperparams(

##                 theta,

##                 hyperparams_value[0], hyperparams_value[1],

##                 hyperparams_value[2], hyperparams_value[3]

##             )

## 
##             s_sample[i, :] = s

##             theta_sample[i, :] = theta

##             hyperparams_sample[i, :] = hyperparams_value

## 
##         return hyperparams_sample, s_sample, theta_sample

## 
##     def update_s(self, s: np.ndarray, m: float, M: float,

##                  B: float) -> np.ndarray:

##         """

##         The gibbs sampler for updating s the membership array.

##         """

## 
##         for i in range(len(s)):

##             s_minus = np.delete(s, i)

##             y_minus = np.delete(self.y, i)

## 
##             # the unique components id excluding y_i

##             components_minus = np.unique(s_minus)

## 
##             n_j_minus, v_j_minus, m_j_minus = \

##                 self.calculate_n_v_m_at_j(

##                     components_minus, s_minus, m, B, y_minus)

## 
##             weights_at_j = (n_j_minus *

##                             self.dnorm(self.y[i],

##                                        mean=m_j_minus,

##                                        std=np.sqrt(v_j_minus + self.sigma2)))

## 
##             weights_at_new = M * self.dnorm(

##                 self.y[i], mean=m, std=np.sqrt(B + self.sigma2))

##             weights = (np.append(weights_at_j, weights_at_new) /

##                        (weights_at_j.sum() + weights_at_new))

## 
##             s_i = self.sample_by_prob(np.append(components_minus,

##                                                 components_minus.max() + 1),

##                                       prob=weights)

##             s[i] = s_i

## 
##         # after update s, rearrange the id

##         s = self.rearrange_s(s)

##         return s

## 
##     def update_theta(self, theta: np.ndarray, s: np.ndarray, m: float,

##                      B: float) -> np.ndarray:

##         """

##         The gibbs sampler for updating theta the components center array.

##         """

##         components = np.unique(s)

##         theta_star = np.zeros_like(components, dtype=np.float64)

## 
##         for i in range(len(components)):

##             sum_y_j = self.y[s == components[i]].sum()

##             n_j = (s == components[i]).sum()

##             theta_mean = (self.sigma2 * m + B * sum_y_j) / (self.sigma2 +

##                                                             B * n_j)

## 
##             theta_var = (self.sigma2 * B) / (self.sigma2 + B * n_j)

##             theta_star[i] = np.random.normal(loc=theta_mean,

##                                              scale=np.sqrt(theta_var))

##             theta[s == components[i]] = theta_star[i]

## 
##         return theta

## 
##     def update_hyperparams(self, theta: np.ndarray, m: float, M: float,

##                            B: float, eta: float) -> np.ndarray:

##         """

##         The gibbs sampler for updating the hyperparams.

##         """

## 
##         theta_star = np.unique(theta)

##         k = len(theta_star)

## 
##         # update m

##         D1 = 1 / (1 / self.D + k / B)

##         m1 = D1 * (self.m0 / self.D + theta_star.sum() / B)

##         m = np.random.normal(m1, np.sqrt(D1))

## 
##         # update B

##         a1 = self.a + k / 2

##         b1 = self.b + ((theta_star - m)**2).sum() / 2

##         B = 1 / np.random.gamma(shape=a1, scale=1 / b1)

## 
##         # update eta

##         eta = np.random.beta(M + 1, self.y.shape[0])

## 
##         # update M

##         weight = ((self.c + k - 1) / (self.c + k - 1 + self.y.shape[0] *

##                                       (self.d - np.log(eta))))

##         unif = np.random.rand()

##         if unif < weight:

##             c1 = self.c + k

##             d1 = self.d - np.log(eta)

##             M = np.random.gamma(shape=c1, scale=1 / d1)

## 
##         else:

##             c1 = self.c + k - 1

##             d1 = self.d - np.log(eta)

##             M = np.random.gamma(shape=c1, scale=1 / d1)

## 
##         return np.array([m, M, B, eta])

## 
##     def calculate_n_v_m_at_j(self, components: np.ndarray, s: np.ndarray,

##                              m: np.ndarray, B: np.ndarray, y: np.ndarray):

##         """

##         calculate some important statistics for updating s.

##         especially the mean(`m_j`) and variance(`v_j`)

##             of center for each components

##         (excluding the ith data point).

##         """

## 
##         # the number of observations in each components

##         n_j = np.array([(s == i).sum() for i in components])

## 
##         sum_of_y_in_group_j = np.array([y[s == i].sum() for i in components])

## 
##         v_j = 1 / (1 / B + n_j / self.sigma2)

##         m_j = v_j * (m / B + sum_of_y_in_group_j / self.sigma2)

## 
##         return n_j, v_j, m_j

## 
##     def dnorm(self, x: float, mean: np.ndarray, std: np.ndarray) -> float:

##         """

##         since numba.jitclass does not support scipy.stat

##         a simple function to calculate the normal density

##         """

##         return np.exp(-((x - mean) / std)**2 / 2) / (std * np.sqrt(2 * np.pi))

## 
##     def sample_by_prob(self, x: np.ndarray, prob: np.ndarray) -> int:

##         """

##         since numba.jitclass does not support np.random.choice

##             with argument `p`.

##         a simple function to implement sample with probability.

##         """

##         cumprob = np.cumsum(prob)

##         unif = np.random.rand()

##         y = x[-1]

##         for i in range(len(cumprob)):

##             if unif < cumprob[i]:

##                 y = x[i]

##                 break

##         return y


## ---- echo = FALSE, eval = F--------------------------------------------------
## rmarkdown::render("hw1.rmd")

