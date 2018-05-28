import itertools
import numpy as np
from copy import deepcopy
import time
import sys
import math



def transition(s, a, demand, LT_s, LT_f, h, b, C_s, C_f,Inv_Max,Inv_Min, cap_fast, cap_slow):
    done = False
    s1 = deepcopy(s)
    reward = 0
    s1[LT_f] += a[0]
    s1[LT_s] += a[1]
    s1[0] += - demand
    reward += math.ceil(a[0]/cap_fast) * C_f +math.ceil(a[1]/cap_slow) * C_s
    if s1[0] >= 0:
        reward += s1[0] * h
    else:
        reward += -s1[0] * b
    s1[0] += s1[1]
    for i in range(1, LT_s):
        s1[i] = s1[i + 1]
    s1[LT_s] = 0
    if (s1[0] > Inv_Max):
        s1[0] = Inv_Max
        done = True
    if s1[0] < Inv_Min:
        s1[0] = Inv_Min
        done = True
    return reward / 1000000, s1, done

def CreateStates(LT_f, LT_s, Inv_Max, Inv_Min, O_f, O_s):
    Temp = []
    total_pipe = []
    total_pipe.append(range(Inv_Min, Inv_Max + 1))
    for i in range(1, LT_f + 1):
        total_pipe.append(range(O_f + O_s + 1))
    for i in range(LT_f + 1, LT_s):
        total_pipe.append(range(O_s + 1))
    for index, i in enumerate(itertools.product(*total_pipe)):
        Temp.append(list(i))
        Temp[index].append(0)
    return np.array(Temp)


def CreateActions(OrderFast, OrderSlow):
    Temp = [0 for z in range((OrderFast + 1) * (OrderSlow + 1))]
    for index, i in enumerate(itertools.product(list(range(0, OrderFast + 1)), list(range(0, OrderSlow + 1)))):
        Temp[index] = i
    return np.array(Temp)


def to_string(state):
    s = ""
    for element in state:
        s += str(element) + "/"
    return s


def CreateDictStates(NewStates):
    dict_states = {}
    for index, state in enumerate(NewStates):
        s = to_string(state)
        dict_states[s] = index
    return dict_states


def action_TBS(F,Q,state):
    temp = np.zeros(2)
    temp[1] = Q
    temp[0] = max(0,F - (np.sum(state)+Q))
    return temp

def action_SIDB(F,S,state):
    temp = np.zeros(2)
    temp[0] = max(0,F - np.sum(state))
    temp[1] = max(0,S - (np.sum(state) + temp[0]))
    return temp

def action_DIDB(F,S,state,LT_f,LT_s):
    temp = np.zeros(2)
    temp[0] = max(0,F - np.sum(state[0:LT_f+1]))
    temp[1] = max(0,S - (np.sum(state) + temp[0]))
    return temp

def action_CDIDB(F,S,state,Cap,LT_f,LT_s):
    temp = np.zeros(2)
    temp[0] = max(0,F - np.sum(state[0:LT_f+1]))
    temp[1] = min(Cap, S - (np.sum(state) + temp[0]))
    temp[1] = max(0,temp[1])
    #print(state,F,S,Cap,temp)
    return temp

def BenchmarkPolicy(States, F ,S = None, Q=None,Cap = None, LT_f=None, LT_s=None,  TBS = None, SIDB = None,DIDB = None, CDIDB = None):
    policy = np.array([len(States),len(States)])
    for state in States:
        if(TBS):
            policy[state][action_TBS(F,Q,state)] = 1
        if(SIDB):
            policy[state][action_SIDB(F, S, state)] = 1
        if(DIDB):
            policy[state][action_DIDB(F, S, state,LT_f,LT_s)] = 1
        if(CDIDB):
            policy[state][action_CDIDB(F, S, state,Cap,LT_f,LT_s)] = 1
    return policy
