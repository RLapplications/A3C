import threading
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import randint
from time import sleep
from copy import deepcopy
import os
import itertools
import csv
import time
import random
import argparse
Demand = []

with open("./Demand100000.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        Demand.append(int(row[0]))

def new_transition(s, a, demand, LT_s, LT_f, h, b, C_s, C_f, Inv_Max, Inv_Min):
    done = False
    s1 = deepcopy(s)
    reward = 0
    s1[0] += - demand
    s1[LT_f] += a[0]
    s1[LT_s] += a[1]
    reward += a[0] * C_f + a[1] * C_s

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
    if s1[0] >= 0:
        reward += s1[0] * h
    else:
        reward += -s1[0] * b
    return -reward / 1000000, s1, done



# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def CreateActions(OrderFast, OrderSlow):
    Temp = [0 for z in range((OrderFast + 1) * (OrderSlow + 1))]
    z = 0
    for i in itertools.product(list(range(0, OrderFast + 1)), list(range(0, OrderSlow + 1))):
        Temp[z] = i
        z += 1
    return Temp


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor):
        with tf.variable_scope(scope):
            self.entropy_factor = entropy_factor
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

            if depth_nn_hidden >= 1:
                self.hidden1 = slim.fully_connected(inputs=self.inputs, num_outputs=depth_nn_layers_hidden[0],
                                                    activation_fn=activation_nn_hidden[0])
                self.state_out = slim.fully_connected(inputs=self.hidden1, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 2:
                self.hidden2 = slim.fully_connected(inputs=self.hidden1, num_outputs=depth_nn_layers_hidden[1],
                                                    activation_fn=activation_nn_hidden[1])
                self.state_out = slim.fully_connected(inputs=self.hidden2, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 3:
                self.hidden3 = slim.fully_connected(inputs=self.hidden2, num_outputs=depth_nn_layers_hidden[2],
                                                    activation_fn=activation_nn_hidden[2])
                self.state_out = slim.fully_connected(inputs=self.hidden3, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 4:
                self.hidden4 = slim.fully_connected(inputs=self.hidden3, num_outputs=depth_nn_layers_hidden[3],
                                                    activation_fn=activation_nn_hidden[3])
                self.state_out = slim.fully_connected(inputs=self.hidden4, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(self.state_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)



            self.value = slim.fully_connected(self.state_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy =  -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.policy_loss = tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * self.entropy_factor

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, best_path, global_episodes,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.best_path = best_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.no_improvement = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.no_improvement_increment = self.no_improvement.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(
            model_path + str(self.number) + str(time.strftime(" %Y%m%d-%H%M%S")))
        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor)
        self.update_local_ops = update_target_graph('global', self.name)
        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        self.bool_evaluating = None
        self.best_solution = 9999999999999

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages  # ,
                     # self.local_AC.state_in[0]:self.batch_rnn_state[0],
                     # self.local_AC.state_in[1]:self.batch_rnn_state[1]
                     }

        v_l, p_l, e_l, g_n, v_n, Policy, _ = sess.run(
            [self.local_AC.value_loss,  # self.batch_rnn_state REMOVED
             self.local_AC.policy_loss,
             self.local_AC.entropy,
             self.local_AC.grad_norms,
             self.local_AC.var_norms,
             # self.local_AC.state_out,
             self.local_AC.policy,
             self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver, saver_best,Demand, LT_s, LT_f, h, b, C_s, C_f, InvMax,
             InvMin,initial_state,Penalty,Demand_Max,max_training_episodes,actions,p_len_episode_buffer,max_no_improvement,pick_largest,verbose,entropy_decay,entropy_min):
        episode_count = sess.run(self.global_episodes)

        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while episode_count < max_training_episodes:  # not coord.should_stop():
                if (episode_count % 50 == 0):
                    self.bool_evaluating = True
                else:
                    self.bool_evaluating = None
                if (episode_count % 10 == 0 and self.local_AC.entropy_factor > entropy_min):
                    self.local_AC.entropy_factor *= entropy_decay
                    print('CHECK',self.local_AC.entropy_factor)
                sess.run(self.update_local_ops)
                eval_performance = []
                for i in range(10):
                    episode_buffer = []
                    episode_values = []

                    eval_buffer = []
                    episode_reward = 0
                    episode_step_count = 0
                    d = False

                    if self.bool_evaluating == True:
                        self.inv_vect = np.array(initial_state)
                    else:
                        self.inv_vect = np.array(
                            initial_state)
                    s = deepcopy(self.inv_vect)
                    trained = False

                    while (d == False and episode_step_count < max_episode_length - 1):  # modified to continue looping
                            # Take an action using probabilities from policy network output.
                            a_dist, v = sess.run([self.local_AC.policy, self.local_AC.value],
                                                 feed_dict={self.local_AC.inputs: [s]})  # ,

                            if  pick_largest:# or self.bool_evaluating:
                                a = np.argmax(a_dist[0])
                            else:
                                a = np.random.choice(np.arange(len(a_dist[0])), p=a_dist[0])

                            if self.bool_evaluating == True:
                                r,s1,d = new_transition(s, actions[a], Demand[episode_step_count*(i+1)], LT_s, LT_f, h, b, C_s, C_f, InvMax, InvMin)
                                d = False

                            else:
                                r, s1, d = new_transition(s, actions[a], random.randint(0, Demand_Max), LT_s, LT_f, h, b, C_s,
                                                          C_f, InvMax, InvMin)
                                d = False

                            if self.bool_evaluating == True:
                                eval_buffer.append([s, actions[a], r, s1, d, v[0, 0]])
                            episode_buffer.append([s, a, r, s1, d, v[0, 0]])

                            episode_values.append(v[0, 0])
                            episode_reward += r
                            s = deepcopy(s1)
                            episode_step_count += 1

                            # If the episode hasn't ended, but the experience buffer is full, then we
                            # make an update step using that experience rollout.
                            if len(episode_buffer) == p_len_episode_buffer and d != True and episode_step_count != max_episode_length - 1 and self.bool_evaluating != True:
                                # Since we don't know what the true final return is, we "bootstrap" from our current
                                # value estimation.
                                v1 = sess.run(self.local_AC.value,
                                              feed_dict={self.local_AC.inputs: [s]})[0, 0]  # ,

                                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                                trained =  True
                                #print(v_l,p_l,e_l,g_n,v_n)
                                #print("v_l", v_l, v_l / (v_l - p_l - self.local_AC.entropy_factor * e_l), "p_l", p_l,
                                #      -p_l / (v_l - p_l - self.local_AC.entropy_factor * e_l), "e_l",
                                #      self.local_AC.entropy_factor * e_l,
                                #      self.local_AC.entropy_factor * (-e_l) / (v_l - p_l - self.local_AC.entropy_factor * e_l),
                                #      "g_n", g_n, "v_n", v_n)
                                episode_buffer = []
                                sess.run(self.update_local_ops)

                    if(self.bool_evaluating != True):
                        break
                    else:
                        eval_performance.append(episode_reward/episode_step_count)
                        #print(i,eval_performance)
                if(self.bool_evaluating):
                    #print('PRIOR',eval_performance)
                    mean_performance = np.mean(eval_performance)
                    #print('CHECK',mean_performance)

                if self.bool_evaluating == True:
                    #if(verbose): print("EVALUATION", episode_reward / episode_step_count, episode_step_count)
                    if (verbose): print("EVALUATION",mean_performance)
                    if (mean_performance < self.best_solution):# and episode_step_count == max_episode_length - 1):
                        self.best_solution = mean_performance#episode_reward / episode_step_count
                        f= open(self.best_path +"/best_solution%i.txt"%self.number,"w")
                        f.write(str(self.best_solution) + ' ' + str (episode_step_count))
                        f.close()
                        saver_best.save(sess, self.best_path + '/Train_' + str(self.number) + '/model_' + ' ' + str(
                            episode_count) + '.cptk')
                        sess.run(self.no_improvement.assign(0))
                        # print(sess.run(self.number,self.no_improvement))

                if self.bool_evaluating != True:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_step_count)
                    self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                #if len(episode_buffer) != 0 and d == True:
                #    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, Penalty)
                #    trained = True
                    #rint("CHECK:  v_l", v_l, v_l / (v_l - p_l - self.local_AC.entropy_factor * e_l), "p_l", p_l,
                    #      -p_l / (v_l - p_l - self.local_AC.entropy_factor * e_l), "e_l",
                    #      self.local_AC.entropy_factor * e_l,
                    #      self.local_AC.entropy_factor * (-e_l) / (v_l - p_l - self.local_AC.entropy_factor * e_l),
                    #      "g_n", g_n, "v_n", v_n)
                   # print(d)
                #elif len(episode_buffer) != 0:
                #    v1 = sess.run(self.local_AC.value,
                #                  feed_dict={self.local_AC.inputs: [s]})[0, 0]  # ,
                #
                #    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                #    print("CHECK:  v_l",v_l,v_l/(v_l-p_l-self.local_AC.entropy_factor*e_l),"p_l",p_l,-p_l/(v_l-p_l-self.local_AC.entropy_factor*e_l),"e_l",self.local_AC.entropy_factor*e_l,self.local_AC.entropy_factor*(-e_l)/(v_l-p_l-self.local_AC.entropy_factor*e_l),"g_n",g_n,"v_n",v_n)

                # Periodically save gifs of episodes, model parameters, and summary statistics.


                if episode_count % 250 == 0 and self.name == 'worker_0':
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    # print("Saved Model")
                if self.bool_evaluating != True:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])


                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward per period', simple_value=float(mean_reward / mean_length))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))

                    self.summary_writer.add_summary(summary, episode_count)
                    #tf.summary.FileWriter(model_path, sess.graph)
                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                sess.run(self.no_improvement_increment)

                episode_count += 1
                if sess.run(self.no_improvement) >= max_no_improvement:
                    break


def NewCreateStates(LT_f,LT_s,Inv_Max,Inv_Min,O_f,O_s):
    Temp = []
    total_pipe = []
    total_pipe.append(range(Inv_Min,Inv_Max+1))
    for i in range(1,LT_f+1):
        total_pipe.append(range(O_f+O_s+1))
    for i in range(LT_f+1,LT_s):
        total_pipe.append(range(O_s+1))
    for index,i in enumerate(itertools.product(*total_pipe)):
        Temp.append(list(i))
        Temp[index].append(0)
    return Temp

def write_parameters(model_path, depth_nn_hidden, depth_nn_layers_hidden, depth_nn_out, entropy_factor,
                     activation_nn_hidden, activation_nn_out, learning_rate, optimizer, activations,
                     p_len_episode_buffer, max_episode_length, OrderFast, OrderSlow, LT_s, LT_f, InvMax,
                     max_training_episodes,h,b,C_f,C_s,InvMin,Penalty,initial_state,nb_workers,cap_fast,cap_slow):
    f = open(model_path + "/Parameters.txt", "w")
    f.write("depth_nn_hidden: " + str(depth_nn_hidden))
    f.write("\ndepth_nn_layers_hidden " + str(depth_nn_layers_hidden))
    f.write("\ndepth_nn_out: " + str(depth_nn_out))
    f.write("\nentropy_factor " + str(entropy_factor))
    f.write("\nactivation_nn_hidden: " + str(activation_nn_hidden))
    f.write("\nactivation_nn_out " + str(activation_nn_out))
    f.write("\nLearning Rate: " + str(learning_rate))
    f.write("\noptimizer " + str(optimizer))
    f.write("\nactivations: " + str(activations))
    f.write("\np_len_episode_buffer " + str(p_len_episode_buffer))
    f.write("\nmax_episode_length: " + str(max_episode_length))
    f.write("\nOrderFast " + str(OrderFast))
    f.write("\nOrderSlow " + str(OrderSlow))
    f.write("\nLT_s " + str(LT_s))
    f.write("\nLT_f " + str(LT_f))
    f.write("\nh " + str(h))
    f.write("\nb " + str(b))
    f.write("\nC_f " + str(C_f))
    f.write("\nC_s " + str(C_s))
    f.write("\nInvMin " + str(InvMin))
    f.write("\nInvMax " + str(InvMax))
    f.write("\nPenalty " + str(Penalty))
    f.write("\ninitial_state " + str(initial_state))
    f.write("\nmax_training_episodes " + str(max_training_episodes))
    f.write("\nnb_workers" + str(nb_workers))
    f.write("\ncap_fast" + str(cap_fast))
    f.write("\ncap_slow" + str(cap_slow))
    f.close()
    return





def objective(args):

    learning_rate = args.initial_lr
    entropy_factor = args.entropy
    gamma = args.gamma
    max_no_improvement = args.max_no_improvement
    max_training_episodes = args.max_training_episodes
    depth_nn_hidden = args.depth_nn_hidden
    depth_nn_layers_hidden = args.depth_nn_layers_hidden
    depth_nn_out = args.depth_nn_out
    p_len_episode_buffer = args.p_len_episode_buffer
    initial_state = args.initial_state
    LT_s = 3
    initial_state=initial_state*LT_s
    initial_state.append(0)

    InvMax = args.invmax  # (LT_s+1)*(2*Demand_Max+1)
    InvMin = args.invmin  # -(LT_s+1)*(2*Demand_Max)
    training = args.training
    pick_largest = args.high
    nb_workers = args.nbworkers
    verbose = args.verbose
    entropy_decay = args.entropy_decay
    entropy_min = args.entropy_min
    activation_nn_hidden = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
    activation_nn_out = tf.nn.relu
    optimizer = tf.train.AdamOptimizer(learning_rate)
    activations = [tf.nn.relu, tf.nn.relu]
    max_episode_length = 100

    # discount rate for advantage estimation and reward discounting


    Demand_Max = 4
    OrderFast = 5
    OrderSlow = 5

    Penalty = 1


    LT_f = 0


    h = -5
    b = -495
    C_f = -150
    C_s = -100
    cap_fast = 1
    cap_slow = 1


    max_training_episodes = 10000000

    actions = CreateActions(OrderFast, OrderSlow)  # np.array([[0,0],[0,5],[5,0],[5,5]])
    a_size = len(actions)  # Agent can move Left, Right, or Fire
    s_size = LT_s + 1

    tf.reset_default_graph()
    if training:
        load_model = False
    else:
        load_model=True

    model_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/model'
    best_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/best'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    # no_improvement = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = optimizer  # tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network = AC_Network(s_size, a_size, 'global', None,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor)  # Generate global network
    num_workers = nb_workers  # multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    workers = []

    write_parameters(model_path, depth_nn_hidden, depth_nn_layers_hidden, depth_nn_out, entropy_factor,
                     activation_nn_hidden, activation_nn_out, learning_rate, optimizer, activations,
                     p_len_episode_buffer, max_episode_length, OrderFast, OrderSlow, LT_s, LT_f, InvMax,
                     max_training_episodes,h,b,C_f,C_s,InvMin,Penalty,initial_state,nb_workers,cap_fast,cap_slow)

    # Create worker classes
    for i in range(num_workers):
        if not os.path.exists(best_path + '/Train_' + str(i)):
            os.makedirs(best_path + '/Train_' + str(i))
        workers.append(Worker(i, s_size, a_size, trainer, model_path, best_path, global_episodes,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor))
    saver = tf.train.Saver(max_to_keep=5)
    saver_best = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:

            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state('./')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.

        if(training):
            worker_threads = []
            temp_best_solutions = np.zeros(len(workers))
            for worker in workers:
                worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver, saver_best,Demand, LT_s, LT_f, h, b, C_s, C_f, InvMax, InvMin,initial_state,Penalty,Demand_Max,max_training_episodes,actions,p_len_episode_buffer,max_no_improvement,pick_largest,verbose,entropy_decay,entropy_min)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)
            for index, worker in enumerate(workers):
                temp_best_solutions[index] = worker.best_solution
            best_solution_found = np.min(temp_best_solutions)
        else:
            States = NewCreateStates(LT_f,LT_s,10,-10,OrderFast,OrderSlow)
            print(States)
            policy_fast = []
            policy_slow = []
            A3C_policy = []
            for index, state in enumerate(States):
                prob_vector = sess.run(workers[0].local_AC.policy,feed_dict={workers[0].local_AC.inputs:[state]})[0]
                A3C_policy.append(prob_vector)
                #print(state,prob_vector,np.sum(prob_vector))
                action_prob_fast = np.zeros(OrderFast + 1)
                action_prob_slow = np.zeros(OrderSlow + 1)

                for i in range(len(actions)):
                    action_prob_fast[actions[i][0]] += prob_vector[i]
                    action_prob_slow[actions[i][1]] += prob_vector[i]
                print(state,np.argmax(action_prob_fast),np.argmax(action_prob_slow),"FAST", action_prob_fast,"SLOW", action_prob_slow)
                policy_fast.append(deepcopy(action_prob_fast))
                policy_slow.append(deepcopy(action_prob_slow))

            np.savetxt('A3C_policy.csv',A3C_policy,delimiter=';')
            with open('cost.csv', 'w') as f:
               for index, i in enumerate(States):
                   for j in States[index]:
                        f.write(str(j) + ';')
                   f.write(';')
                   for j in policy_fast[index]:
                        f.write(str(j) + ';')
                   f.write(';')
                   for j in policy_slow[index]:
                        f.write(str(j) + ';')
                   f.write(';')
                   f.write('\n')
        return



def obj_bo(list):
    learning_rate = list[0]
    entropy_factor = list[1]
    gamma = list[2]
    max_no_improvement = 250
    max_training_episodes = 500000
    depth_nn_hidden = list[2]
    depth_nn_layers_hidden = [list[3],list[4],list[5],list[6]]
    depth_nn_out = list[7]
    p_len_episode_buffer = list[8]#30
    initial_state = [3]
    LT_s = 1
    initial_state = initial_state * LT_s
    initial_state.append(0)
    InvMax = list[9]#10
    InvMin = list[10]#-10  # -(LT_s+1)*(2*Demand_Max)
    training = True
    pick_largest = False
    verbose = False
    activations = [tf.nn.relu, tf.nn.sigmoid,tf.nn.elu]
    activation_nn_hidden = [activations[list[11]], activations[list[12]], activations[list[13]], activations[list[14]]]
    activation_nn_out = activations[list[15]]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    verbose = True
    entropy_decay = 0.9
    entropy_min = 1
    max_episode_length = 100

    # discount rate for advantage estimation and reward discounting
    nb_workers = 4

    Demand_Max = 4
    OrderFast = 5
    OrderSlow = 5

    Penalty = -1

    LT_f = 0

    h = -5
    b = -495
    C_f = -150
    C_s = -100
    cap_fast = 1
    cap_slow = 1


    actions = CreateActions(OrderFast, OrderSlow)  # np.array([[0,0],[0,5],[5,0],[5,5]])
    a_size = len(actions)  # Agent can move Left, Right, or Fire
    s_size = LT_s + 1

    tf.reset_default_graph()
    if training:
        load_model = False
    else:
        load_model = True

    model_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/model'
    best_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/best'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    # no_improvement = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = optimizer  # tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network = AC_Network(s_size, a_size, 'global', None, depth_nn_out, activation_nn_hidden, depth_nn_hidden,
                                depth_nn_layers_hidden, activation_nn_out, entropy_factor)  # Generate global network
    num_workers = nb_workers  # multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    workers = []

    write_parameters(model_path, depth_nn_hidden, depth_nn_layers_hidden, depth_nn_out, entropy_factor,
                     activation_nn_hidden, activation_nn_out, learning_rate, optimizer, activations,
                     p_len_episode_buffer, max_episode_length, OrderFast, OrderSlow, LT_s, LT_f, InvMax,
                     max_training_episodes, h, b, C_f, C_s, InvMin, Penalty, initial_state, nb_workers, cap_fast,
                     cap_slow)

    # Create worker classes
    for i in range(num_workers):
        if not os.path.exists(best_path + '/Train_' + str(i)):
            os.makedirs(best_path + '/Train_' + str(i))
        workers.append(Worker(i, s_size, a_size, trainer, model_path, best_path, global_episodes, depth_nn_out,
                              activation_nn_hidden, depth_nn_hidden, depth_nn_layers_hidden, activation_nn_out,
                              entropy_factor))
    saver = tf.train.Saver(max_to_keep=5)
    saver_best = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:

            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state('./')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.

        if (training):
            worker_threads = []
            temp_best_solutions = np.zeros(len(workers))
            for worker in workers:
                worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver, saver_best, Demand,
                                                  LT_s, LT_f, h, b, C_s, C_f, InvMax, InvMin, initial_state, Penalty,
                                                  Demand_Max, max_training_episodes, actions, p_len_episode_buffer,
                                                  max_no_improvement, pick_largest,verbose,entropy_decay,entropy_min)
                t = threading.Thread(target=(worker_work))
                t.start()
                sleep(0.5)
                worker_threads.append(t)
            coord.join(worker_threads)
            for index, worker in enumerate(workers):
                temp_best_solutions[index] = worker.best_solution
            best_solution_found = np.min(temp_best_solutions)
        else:
            States = NewCreateStates(LT_f, LT_s, 10, -10, OrderFast, OrderSlow)
            print(States)
            policy_fast = []
            policy_slow = []
            A3C_policy = []
            for index, state in enumerate(States):
                prob_vector = sess.run(workers[0].local_AC.policy, feed_dict={workers[0].local_AC.inputs: [state]})[0]
                A3C_policy.append(prob_vector)
                # print(state,prob_vector,np.sum(prob_vector))
                action_prob_fast = np.zeros(OrderFast + 1)
                action_prob_slow = np.zeros(OrderSlow + 1)

                for i in range(len(actions)):
                    action_prob_fast[actions[i][0]] += prob_vector[i]
                    action_prob_slow[actions[i][1]] += prob_vector[i]
                print(state, np.argmax(action_prob_fast), np.argmax(action_prob_slow), "FAST", action_prob_fast, "SLOW",
                      action_prob_slow)
                policy_fast.append(deepcopy(action_prob_fast))
                policy_slow.append(deepcopy(action_prob_slow))

            np.savetxt('A3C_policy.csv', A3C_policy, delimiter=';')
            with open('cost.csv', 'w') as f:
                for index, i in enumerate(States):
                    for j in States[index]:
                        f.write(str(j) + ';')
                    f.write(';')
                    for j in policy_fast[index]:
                        f.write(str(j) + ';')
                    f.write(';')
                    for j in policy_slow[index]:
                        f.write(str(j) + ';')
                    f.write(';')
                    f.write('\n')
        return best_solution_found

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-lr', '--initial_lr', default=0.0001, type=float,
                        help="Initial value for the learning rate. If a value of 0 is specified, the learning rate will be sampled from a LogUniform(10**-4, 10**-2) distribution. Default = 0.001",
                        dest="initial_lr")

    parser.add_argument('--entropy', default=0.000005, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic). Default = 0.01",
                        dest="entropy")

    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor. Default = 0.99", dest="gamma")

    parser.add_argument('--max_no_improvement', default=50000, type=float, help="max_no_improvement. Default = 5000", dest="max_no_improvement")

    parser.add_argument('--max_training_episodes', default=10000000, type=float, help="max_training_episodes. Default = 10000000",
                        dest="max_training_episodes")

    parser.add_argument('--depth_nn_hidden', default=3, type=float,
                        help="depth_nn_hidden. Default = 3",
                        dest="depth_nn_hidden")

    parser.add_argument('--depth_nn_out', default=20, type=float,
                        help="depth_nn_out. Default = 20",
                        dest="depth_nn_out")

    parser.add_argument('--depth_nn_layers_hidden', default=[150,120,80,40], type=float,
                        help="depth_nn_layers_hidden. Default = [40,20,10,10]",
                        dest="depth_nn_layers_hidden")


    parser.add_argument('--p_len_episode_buffer', default=3, type=float,
                        help="p_len_episode_buffer. Default = 50",
                        dest="p_len_episode_buffer")

    parser.add_argument('--initial_state', default=[3], type=float,
                        help="initial_state. Default = [3,0]",
                        dest="initial_state")


    parser.add_argument('--invmax', default=40, type=float,
                        help="invmax. Default = 150",
                        dest="invmax")

    parser.add_argument('--invmin', default=-40 , type=float,
                        help="invmin. Default = -15",
                        dest="invmin")

    parser.add_argument('--training', default= True, type=str,
                        help="training. Default = True",
                        dest="training")
    parser.add_argument('--high', default= False, type=float,
                        help="Pick largest likelihood. Default = False",
                        dest="high")
    parser.add_argument('--nbworkers', default= 4, type=float,
                        help="Number of A3C workers. Default = 4",
                        dest="nbworkers")
    parser.add_argument('--verbose', default= True, type=str,
                        help="Print evaluation results. Default = False",
                        dest="verbose")

    parser.add_argument('--entropy_decay', default= 0.9, type=float,
                        help="Entropy_decay. Default = 0.95",
                        dest="entropy_decay")

    parser.add_argument('--entropy_min', default= 1, type=float,
                        help="entropy_min. Default = 0",
                        dest="entropy_min")
    args = parser.parse_args()

    objective(args)