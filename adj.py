# %%
import timeit
import numba  # type: ignore
import numpy as np  # type: ignore
# %%

# %%


@numba.njit
def get_adj(comps: np.ndarray) -> np.ndarray:

    ncol = comps.shape[1]
    nrow = comps.shape[0]
    adj_sum = np.zeros((ncol, ncol))

    for i in range(nrow):
        for j in range(ncol):
            for k in range(ncol):
                if comps[i, j] == comps[i, k]:
                    adj_sum[j, k] += 1
    return adj_sum


def get_adj_nojit(comps: np.ndarray) -> np.ndarray:
    ncol = comps.shape[1]
    nrow = comps.shape[0]
    adj_sum = np.zeros((ncol, ncol))

    for i in range(nrow):
        for j in range(ncol):
            for k in range(ncol):
                if comps[i, j] == comps[i, k]:
                    adj_sum[j, k] += 1
    return adj_sum


# %%
comps = np.random.choice(range(10), size=(1000, 1500))

start = timeit.default_timer()

get_adj(comps)

stop = timeit.default_timer()
print('Time: ', stop - start)
# 2.58s

# %%
# %%


start = timeit.default_timer()

get_adj_nojit(comps)

stop = timeit.default_timer()
print('Time: ', stop - start)

# 524s
# %%
