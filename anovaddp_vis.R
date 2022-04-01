# %%
library(tidyverse)

y_v1w1 <- read_csv("y_v1w1.csv") |>
    rename(x = `0`, y = `1`)
y_v2w1 <- read_csv("y_v2w1.csv") |>
    rename(x = `0`, y = `1`)
y_v1w2 <- read_csv("y_v1w2.csv") |>
    rename(x = `0`, y = `1`)
y_v2w2 <- read_csv("y_v2w2.csv") |>
    rename(x = `0`, y = `1`)
# %%
load("ypred.RData")

# ba is short for bayesian anova
y_ba_1 <- y_pred$y.1
y_ba_2 <- y_pred$y.2


y_ba_v1w1_1 <- y_ba_1[, 1]
y_ba_v1w2_1 <- y_ba_1[, 2]
y_ba_v2w1_1 <- y_ba_1[, 3]
y_ba_v2w2_1 <- y_ba_1[, 4]

y_ba_v1w1_2 <- y_ba_2[, 1]
y_ba_v1w2_2 <- y_ba_2[, 2]
y_ba_v2w1_2 <- y_ba_2[, 3]
y_ba_v2w2_2 <- y_ba_2[, 4]

y_ba_v1w1 <- cbind(y_ba_v1w1_1, y_ba_v1w1_2) |>
    as_tibble()
colnames(y_ba_v1w1) <- c("x", "y")

y_ba_v1w2 <- cbind(y_ba_v1w2_1, y_ba_v1w2_2) |>
    as_tibble()
colnames(y_ba_v1w2) <- c("x", "y")

y_ba_v2w1 <- cbind(y_ba_v2w1_1, y_ba_v2w1_2) |>
    as_tibble()
colnames(y_ba_v2w1) <- c("x", "y")

y_ba_v2w2 <- cbind(y_ba_v2w2_1, y_ba_v2w2_2) |>
    as_tibble()
colnames(y_ba_v2w2) <- c("x", "y")
# %%

plot_contour <- function(y_vw, y_ba_vw, title) {
    ggplot(y_vw) +
        stat_density_2d(
            aes(x = x, y = y, fill = ..level..),
            geom = "polygon"
        ) +
        scale_fill_gradient2(
            low = "#FFFAFA", mid = "#BBFFFF", high = "#483D8B"
        ) +
        scale_x_continuous(breaks = seq(-2, 10, by = 2), limits = c(-2, 10)) +
        scale_y_continuous(breaks = seq(6, 20, by = 2), limits = c(6, 20)) +
        labs(title = title) +
        geom_density_2d(
            aes(x = x, y = y),
            color = "red",
            data = y_ba_vw
        )
}

plot_contour_ba <- function(y_ba_vw) {
    ggplot(y_ba_vw) +
        geom_density_2d(
            aes(x = x, y = y),
            color = "red",
        ) +
        scale_x_continuous(breaks = seq(-2, 10, by = 2), limits = c(-2, 10)) +
        scale_y_continuous(breaks = seq(6, 20, by = 2), limits = c(6, 20))
}

# %%

cowplot::plot_grid(
    plot_contour(y_v1w1, y_ba_v1w1, title = "v = 1, w = 1"),
    plot_contour(y_v1w2, y_ba_v1w2, title = "v = 1, w = 2"),
    plot_contour(y_v2w1, y_ba_v2w1, title = "v = 2, w = 1"),
    plot_contour(y_v2w2, y_ba_v2w2, title = "v = 2, w = 2"),
    ncol = 2
)
# %%
