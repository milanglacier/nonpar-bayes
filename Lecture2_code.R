# Thanks for my friend Tamara Broderick for the code

####### Beta distribution

maxiters <- 1000

ex1_draw_betas <- function(a) {
  # Function draw_betas_diffa but with both parameters equal

  ex1_draw_betas_diffa(a, a)
}

ex1_draw_betas_diffa <- function(a1, a2) {
  # Samples and illustrates beta random variables.
  #
  # Args:
  #  a1, a2: the beta parameters
  #
  # Returns:
  #  Nothing.
  #
  # For each press of "enter" samples a beta random variable
  # and illustrates it as a distribution over {1,2}.
  # Press 'x' when finished making samples.

  for (iter in 1:maxiters) {
    # beta random variable sample
    rho <- rbeta(1, a1, a2)

    plot(c(rho, 1 - rho),
      type = "h",
      xlim = c(0, 3),
      ylim = c(0, 1),
      ylab = "frequencies",
      xlab = "index",
      xaxt = "n",
      col = "red",
      lwd = 5,
      main = bquote("(" ~ rho[1] ~ "," ~ rho[2] ~ ")~Beta(" ~ .(a1)
      ~ "," ~ .(a2) ~ ")")
    )
    axis(1,
      at = seq(0, 3, by = 1),
      labels = c("", "1", "2", "")
    )

    # Generate a new draw for each press of "enter"
    # Press 'x' when finished
    line <- readline()
    if (line == "x") {
      return("done")
    }
  }
}

ex1_draw_betas(10)



####### Dirichlet distribution

library(MCMCpack)

# maximum number of Dirichlet random vars to sample
maxiters <- 1000

ex2_draw_diris <- function(K, a_scalar) {
  # function ex2_draw_diris_diffa but with both parameters equal

  ex2_draw_diris_diffa(rep(a_scalar, K))
}

ex2_draw_diris_diffa <- function(a_vector) {
  # Samples and illustrates Dirichlet random variables.
  #
  # Args:
  #  a_vector: the Dirichlet parameters
  #
  # Returns:
  #  Nothing.
  #
  # For each press of "enter" samples a Dirichlet random var
  # and illustrates it as a distribution over {1,2,...,K},
  # where K is the length of the parameter vector.
  # Press 'x' when finished making samples.

  K <- length(a_vector)
  for (iter in 1:maxiters) {
    # Dirichlet random variable sample
    rho <- rdirichlet(1, a_vector)

    # put Dirichlet parameters in the plot title
    a_vector_str <- toString(a_vector)
    plot(as.vector(rho),
      type = "h",
      xlim = c(0, K + 1),
      ylim = c(0, 1),
      ylab = "frequencies",
      xlab = "index",
      xaxt = "n",
      col = "red",
      lwd = 5,
      main = bquote(rho ~ "~Dirichlet(" ~ .(a_vector_str)
      ~ ")")
    )
    axis(1,
      at = seq(0, K + 1, by = 1),
      labels = c("", paste(1:K), "")
    )

    # Generate a new draw for each press of "enter"
    # Press 'x' when finished
    line <- readline()
    if (line == "x") {
      return("done")
    }
  }
}

ex2_draw_diris(4, 1)

#########
# When K>>N
library(MCMCpack)

# maximum number of data points to draw
maxN <- 1000

# use these parameters by default
K_default <- 1000
a_default <- 10 / K_default
# note: default run with these parameters
# appears at the end

ex3_gen_largeK_diri <- function(K, a) {
  # Illustrates cluster assignments using
  # Dirichlet-distributed component probabilities
  #
  # Args:
  #  K: Dirichlet parameter vector length
  #  a: Dirichlet parameter (will be repeated K times)
  #
  # Returns:
  #  Nothing
  #
  # Illustrates Dirichlet-distributed samples
  # as a partition of the unit interval
  # (cf. Kingman paintbox) and illustrates
  # samples from this (random) distribution

  # make the Dirichlet draw
  rhomx <- rdirichlet(1, rep(a, K))
  # various other useful forms of rho
  rho <- as.vector(rhomx)
  rhomxt <- t(rhomx)
  crho <- c(0, cumsum(rho))

  # initialize bar colors so that
  # no components have been chosen yet
  # "grey" = not chosen, "blue" = chosen
  bar_colors <- rep("grey", K)

  # special plot size for very horizontal fig
  x11(width = 8, height = 3)

  for (N in 0:maxN) {
    # want the option to illustrate
    # before draws are made
    if (N > 0) {
      # uniform draw to decide which component is chosen
      u <- runif(1)
      draw <- max(which(crho < u))

      # update bar color of chosen component
      bar_colors[draw] <- "blue"
    }

    # bar plot makes it easy to plot
    # probabilities one after another
    barplot(rhomxt,
      beside = FALSE,
      horiz = TRUE,
      col = bar_colors,
      width = 0.7,
      ylim = c(0, 1),
      main = bquote(rho ~ "~Dirichlet" # ~"("~.(a)~",...,"~.(a)~")"
      ~", K=" ~ .(K) ~ ", N=" ~ .(N))
    )

    if (N > 0) {
      # illustrate the uniform random variable
      points(u, 0.9, pch = 25, col = "red", bg = "red")
    }

    # Generate a new draw for each press of "enter"
    # Press 'x' when finished
    line <- readline()
    if (line == "x") {
      dev.off()
      return("done")
    }
  }
}

# default run with default parameters
ex3_gen_largeK_diri(K_default, a_default)

#####################################
library(MCMCpack)

# maximum number of data points to draw
maxN <- 1000

# use these parameters by default
K_default <- 1000
a_default <- 10 / K_default
# note: default run with these parameters
# appears at the end

ex4_gen_largeK_count <- function(K, a) {
  # Illustrates how number of clusters changes
  # with Dirichlet-distributed component probabilities
  #
  # Args:
  #  K: Dirichlet parameter vector length
  #  a: Dirichlet parmeter (will be repeated K times)
  #
  # Returns:
  #  Nothing

  # make the Dirichlet draw
  rhomx <- rdirichlet(1, rep(a, K))
  # another useful form of rho
  rho <- as.vector(rhomx)

  # records which clusters have been sampled so far
  uniq_draws <- c()
  # cluster samples in order of appearance (ooa)
  ooa_clust <- c()

  for (N in 1:maxN) {
    # draw a cluster assignment from the components
    draw <- sample(1:K, size = 1, replace = TRUE, prob = rho)
    # update info about cluster draws
    uniq_draws <- unique(c(uniq_draws, draw))
    ooa <- which(draw == uniq_draws)
    ooa_clust <- c(ooa_clust, ooa)

    # plot cluster assignments in order of appearance
    plot(seq(1, N),
      ooa_clust,
      xlab = "Sample index",
      ylab = "Cluster by order of appearance",
      ylim = c(0, max(10, length(uniq_draws))),
      xlim = c(0, max(10, N)),
      pch = 19,
      main = bquote(rho ~ "~Dirichlet" # ~"("~.(a)~",...,"~.(a)~")"
      ~", K=" ~ .(K))
    )

    # Generate a new draw for each press of "enter"
    # Press 'x' when finished
    line <- readline()
    if (line == "x") {
      return("done")
    }
  }
}

# default run with default parameters
ex4_gen_largeK_count(K_default, a_default)
