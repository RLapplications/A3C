import New_A3C

import numpy as np
import matplotlib.pyplot as plt

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import os
import time
import csv
import argparse


def main(args):
    space  = [Real(10**-6, 10**-3, "log-uniform", name='initial_lr'),
              Real(10**-7,10**-4, "log-uniform", name='entropy'),
              Categorical([1,2,3,4], name='depth_nn_hidden'),
              Categorical([1, 70, 150], name='depth_nn_layers_hidden[0]'),
              Categorical([1, 50, 70],  name='depth_nn_layers_hidden[1]'),
              Categorical([1, 40, 20],  name='depth_nn_layers_hidden[2]'),
              Categorical([1, 20],  name='depth_nn_layers_hidden[3]'),
              Categorical([1, 8], name='depth_nn_out'),
              Categorical([3, 20],  name='p_len_episode_buffer'),
              Categorical([ 40], name='InvMax'),
              Categorical([-40],  name='invMin'),
              Categorical([0, 1, 2],  name='activation_nn_hidden[0]'),
              Categorical([0,1,2],   name='activation_nn_hidden[1]'),
              Categorical([0,1,2],   name='activation_nn_hidden[2]'),
              Categorical([0,1,2],  name='activation_nn_hidden[3]'),
              Categorical([0,1,2],   name='activation_nn_out'),
              [args.max_no_improvement],
              [args.LT_s],
              [args.cap_slow],
              [args.cap_fast] ]

    log_path = 'BOLogs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    res_gp = gp_minimize(New_A3C.obj_bo, space, n_calls=int(args.iterations), random_state=0,n_jobs=-1)


    print("Best score=%f" % res_gp.fun)

    x_iters = res_gp.x_iters
    func_vals = res_gp.func_vals


    with open(log_path + '/BO_best.csv', 'w') as f:
        f.write(str(res_gp.fun))

    with open(log_path +'/BO_results.csv', 'w') as f:
        for  index,i in enumerate(x_iters):
            for item in i:
                f.write(str(item)+ ';')
            f.write(str(func_vals[index])+';')
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-iterations', '--iterations', default=50, type=float,
                        help="Number of hyperparameter sets tested",
                        dest="iterations")
    parser.add_argument('--max_no_improvement', default=2500, type=float, help="max_no_improvement. Default = 5000",
                        dest="max_no_improvement")
    parser.add_argument('--LT_s', default=4, type=int, help="LT_s. Default = 1", dest="LT_s")
    parser.add_argument('--LT_f', default=0, type=int, help="LT_f. Default = 0",
                        dest="LT_f")
    parser.add_argument('--cap_slow', default=1, type=float,
                        help="cap_slow. Default = 1",
                        dest="cap_slow")
    parser.add_argument('--cap_fast', default=1, type=float,
                        help="cap_fast. Default = 1",
                        dest="cap_fast")

    args = parser.parse_args()
    main(args)