import environment
import time
import argparse
import MarkovChain
import PI
import numpy as np

def DP(args,LT_tested):
    optimal_costs = []
    for i in range(1,LT_tested):
        args.LT_s = i
        States = environment.CreateStates(args.LT_f,args.LT_s,args.Inv_Max,args.Inv_Min,args.OrderFast,args.OrderSlow)
        Actions = environment.CreateActions(args.OrderFast,args.OrderSlow)
        dict_states = environment.CreateDictStates(States)
        optimal_cost = PI.cost_per_period(States,Actions,dict_states, args, i)
        print(optimal_cost)
        optimal_costs.append(optimal_cost)
        np.savetxt('optimal_array.csv',optimal_costs,delimiter=";")
        return optimal_costs




    return
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Demand_Max', '--Demand_Max', default=4, type=float,
                        help="Demand_Max. Default = 4",
                        dest="Demand_Max")
    parser.add_argument('--OrderFast', default=8, type=int,
                        help="OrderFast. Default = 5",
                        dest="OrderFast")
    parser.add_argument('--OrderSlow', default=8, type=int, help="OrderSlow. Default = 5", dest="OrderSlow")
    parser.add_argument('--LT_s', default=4, type=float, help="LT_s. Default = 1", dest="LT_s")
    parser.add_argument('--LT_f', default=0, type=float, help="LT_f. Default = 0",
                        dest="LT_f")
    parser.add_argument('--Inv_Max', default=10, type=float,
                        help="Inv_Max. Default = 10",
                        dest="Inv_Max")
    parser.add_argument('--Inv_Min', default=-10, type=float,
                        help="Inv_Min. Default = -10",
                        dest="Inv_Min")
    parser.add_argument('--cap_slow', default=1, type=float,
                        help="cap_slow. Default = 1",
                        dest="cap_slow")
    parser.add_argument('--cap_fast', default=1, type=float,
                        help="cap_fast. Default = 1",
                        dest="cap_fast")
    parser.add_argument('--C_s', default=100, type=float,
                        help="C_s. Default = 100",
                        dest="C_s")
    parser.add_argument('--C_f', default=150, type=float,
                        help="C_f. Default = 150",
                        dest="C_f")
    parser.add_argument('--h', default=5 , type=float,
                        help="h. Default = 5",
                        dest="h")
    parser.add_argument('--b', default= 495, type=str,
                        help="b. Default = 495",
                        dest="b")
    parser.add_argument('--discount_factor', default= 0.99, type=float,
                        help="discount_factor. Default = 0.99",
                        dest="discount_factor")
    args = parser.parse_args()

    #DP(args,8)
    States = environment.CreateStates(args.LT_f, args.LT_s, args.Inv_Max, args.Inv_Min, args.OrderFast, args.OrderSlow)
    Actions = environment.CreateActions(args.OrderFast, args.OrderSlow)
    dict_states = environment.CreateDictStates(States)

    with open('./A3C_policy.csv') as f:
        #States = []
        policy = []

        for line in f:
            #States.append(line.split(sep=';')[:2])
            policy.append(line.split(sep=';')[0:82])
            #a_prob_s.append(line.split(sep=';')[10:16])
        for index, i in enumerate(policy):
            policy[index] = [float(j) for j in policy[index]]


    print(policy)
    P = PI.TransitionProbs(States, Actions, args.Demand_Max,args.LT_s,args.LT_f,args.h,args.b,args.C_s,args.C_f,args.Inv_Max,args.Inv_Min, args.cap_fast, args.cap_slow,dict_states)
    print('Cost: ',MarkovChain.TestPolicy(States,P,policy))