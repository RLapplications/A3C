import numpy as np
import environment
import MarkovChain
import time

class PI_env():
    def __init__(self, states, actions, P):
        self.nS = len(states)  # len(nA)#len(CreateActions(OrderFast,OrderSlow))
        self.nA = len(actions)  # len(nS)#len(CreateStates())
        self.P = P

def TransitionProbs(States, Actions, Demand_Max,LT_s,LT_f,h,b,C_s,C_f,Inv_Max,Inv_Min, cap_fast, cap_slow,dict_states):
    T = []
    for index, state in enumerate(States):
        Temp1 = []
        for index2, action in enumerate(Actions):
            Temp2 = []
            for index3, demand in enumerate(range(Demand_Max+1)):
                reward, s1, done = environment.transition(state, action, demand, LT_s, LT_f, h, b, C_s, C_f,Inv_Max,Inv_Min, cap_fast, cap_slow)
                Tuple = (1 / (Demand_Max+1), dict_states[environment.to_string(s1)], reward, done)
                Temp2.append(Tuple)
            Temp1.append(Temp2)
        T.append(Temp1)
    return T



def policy_eval(policy, env, discount_factor, theta=0.00000000000001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        # print(delta)
        if delta < theta:
            # print('changed')
            break
    return np.array(V)


def policy_improvement(env, discount_factor,policy_eval_fn=policy_eval):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                # if done != True:
                A[a] += prob * (reward + discount_factor * V[next_state])
                # else:
                #    A[a] += prob * (-99999999999999999999999999999)
        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    counter = 0

    temp = 0
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        print(np.sum(V), temp - np.sum(V))
        temp = np.sum(V)
        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmin(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                # print('check',chosen_a,best_a)
                policy_stable = False
                counter = counter + 1

            policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V

def cost_per_period(States,Actions,dict_states, args, index_LT):
    P = TransitionProbs(States, Actions, args.Demand_Max, args.LT_s, args.LT_f, args.h, args.b, args.C_s, args.C_f,
                           args.Inv_Max, args.Inv_Min, args.cap_fast, args.cap_slow, dict_states)
    start = time.time()

    env = PI_env(States, Actions, P)
    #print('env created', time.time() - start)

    policy, v = policy_improvement(env, args.discount_factor)
    #print(time.time() - start)

    #print("Policy Probability Distribution:")
    #print(policy)
    #print("")

    #print("Value Function:")
    #print(v)
    #print("")
    np.savetxt("policy-LT%i-cap%i.csv" %(index_LT,args.cap_fast), policy, delimiter=";")
    np.savetxt("valuefunction-LT%i-cap%i.csv"%(index_LT,args.cap_fast), v, delimiter=";")
    np.savetxt("States-LT%i-cap%i.csv"%(index_LT,args.cap_fast),States, delimiter=";")
    np.savetxt("Actions-LT%i-cap%i.csv"%(index_LT,args.cap_fast), Actions, delimiter=";")

    #for index, i in enumerate(States):
    #    print(i, Actions[np.argmax(policy[index])])

    MC, MC_R = MarkovChain.MC(States, P, policy)
    steady_state = MarkovChain.steady_state(States, policy, MC)
    #print(steady_state)
    optimal_cost = MarkovChain.cost_steady_state(steady_state, policy, MC, MC_R)
    return optimal_cost