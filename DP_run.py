# %%
import numpy as np
from DP import DPGMM


y = np.append(np.random.normal(loc=16, size=50),
              np.random.normal(loc=8, size=50))
a = DPGMM(y=y,
          sigma2=1.0, m0=2.5, D=1.0, a=1.0, b=1.0, c=1.0, d=1.0,
          initial_components=2, iters=5000)
a.initialize()
hypers, s, theta = a.sample()


# %%
