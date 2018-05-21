import New_A3C

import numpy as np
import matplotlib.pyplot as plt



from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    #boston = load_boston()
    #X, y = boston.data, boston.target
    #n_features = X.shape[1]
#
    ## gradient boosted trees tend to do well on problems like this
    #reg = GradientBoostingRegressor(n_estimators=50, random_state=0)

    from skopt.space import Real, Integer
    from skopt.utils import use_named_args


    # The list of hyper-parameters we want to optimize. For each one we define the bounds,
    # the corresponding scikit-learn parameter name, as well as how to sample values
    # from that dimension (`'log-uniform'` for the learning rate)
    #space  = [Integer(1, 5, name='max_depth'),
    #          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
    #          Integer(1, n_features, name='max_features'),
    #          Integer(2, 100, name='min_samples_split'),
    #          Integer(1, 100, name='min_samples_leaf')]
#
    space  = [Real(10**-2, 10**0, "log-uniform", name='initial_lr'),
              Real(10 ** -1, 10 ** 0, "log-uniform", name='entropy')              ]






    # this decorator allows your objective function to receive a the parameters as
    # keyword arguments. This is particularly convenient when you want to set scikit-learn
    # estimator parameters
    #@use_named_args(space)

    #def objective(**params):
    #    reg.set_params(**params)
    #
    #    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
     #                                   scoring="neg_mean_absolute_error"))


    from skopt import gp_minimize
    res_gp = gp_minimize(New_A3C.obj_bo, space, n_calls=10, random_state=0,n_jobs=-1)

    print("Best score=%f" % res_gp.fun)


    print(res_gp.x_iters)
    print(res_gp.func_vals)
    print(res_gp.specs)
    from skopt.plots import plot_convergence

    plot_convergence(res_gp)