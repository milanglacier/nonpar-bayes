
# %%
library(tidyverse)
library(reticulate)
# %%
# %%
np <- import("numpy")

newy <- np$load("newy.npy")
newy_1 <- newy[, , 1]
newy_2 <- newy[, , 2]

newy_1 <- as_tibble(newy_1)
colnames(newy_1) <- c("v1w1", "v2w1", "v1w2", "v2w2")
newy_1 <- newy_1 |>
    mutate(p = "x")

newy_2 <- as_tibble(newy_2)
colnames(newy_2) <- c("v1w1", "v2w1", "v1w2", "v2w2")
newy_2 <- newy_2 |>
    mutate(p = "y")


newy <- rbind(newy_1, newy_2)
# %%

# first expand new y from a table, shape is 2000 * 5 (column p represent it is y0 or y1,
# the rest 4 dimensional represents
#   the predictive sample when taking 1 level (x) out of 4
#   possible levels of combination at one round of gibbs sampling
# the table will be pivot longer to 8000 * 2
# then pivot wider to
# 4000 * 3, will collapse the column p (takes value x and y)
# to two columns x and y
newy <- newy |>
    pivot_longer(!p, names_to = "(v,w)") |>
    pivot_wider(names_from = p, values_from = value)

# %%

ggplot(newy) +
    stat_density_2d(aes(
        x = x, y = y,
        fill = ..level..
    ),
    geom = "polygon"
    ) +
    scale_fill_gradient2(
        low = "#FFFAFA", mid = "#BBFFFF", high = "#483D8B"
    )

