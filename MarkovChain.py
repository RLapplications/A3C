import environment
import numpy as np
from copy import deepcopy

def MC(states, P, policy):
    MC = np.zeros([len(states),len(states)])
    MC_R = np.zeros([len(states), len(states)])
    for index,action in enumerate(policy):
        for prob, next_state, reward,_ in P[index][np.argmax(action)]:
            MC[index][next_state]+= prob
            MC_R[index][next_state] += reward
    return MC, MC_R

def steady_state(states, policy, MC):
    initial = np.zeros(len(states))
    initialtemp = deepcopy(initial)
    initial[0] = 1

    while not np.array_equal(initial, initialtemp):
        initialtemp = deepcopy(initial)
        initial = np.dot(initial, MC)
    return initial

def cost_steady_state(steady_state, policy, MC, MC_R):
    tempcost = 0
    for index, prob in enumerate(steady_state):
        for index2, prob_next in enumerate(MC[index]):
            tempcost += prob * prob_next * MC_R[index][index2]
    return tempcost

def TestPolicy(states, P, policy):
    MC_m, MC_R = MC(states, P, policy)
    steady = steady_state(states, policy, MC_m)
    cost = cost_steady_state(steady, policy, MC_m,MC_R)
    return cost