import New_A3C

import numpy as np
import matplotlib.pyplot as plt

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

if __name__ == '__main__':

    space  = [Real(10**-5, 10**0, "log-uniform", name='initial_lr'),
              Real(10 ** -20, 10 ** 0, "log-uniform", name='entropy')              ]




    res_gp = gp_minimize(New_A3C.obj_bo, space, n_calls=10, random_state=0,n_jobs=-1)

    print("Best score=%f" % res_gp.fun)


    print(res_gp.x_iters)
    print(res_gp.func_vals)
    print(res_gp.specs)


    plot_convergence(res_gp)