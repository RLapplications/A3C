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

Demand=[]
with open("./Demand.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        Demand.append(int(row[0]))

#HYPERPARAMETERS

#dec_vect = np.zeros(7)





depth_nn_hidden = 1
depth_nn_layers_hidden = [70, 40 , 10 , 20]
depth_nn_out = 40
entropy_factor = 0.0000001
p_len_episode_buffer = 50
gamma = .99
InvMax= 25
learning_rate = 0.0001

activation_nn_hidden =[tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu]
activation_nn_out=tf.nn.relu

optimizer = tf.train.AdamOptimizer(learning_rate)
activations = [tf.nn.relu,tf.nn.relu]


max_episode_length = 1000
 # discount rate for advantage estimation and reward discounting
nb_workers = 4
OrderFast=5
OrderSlow=10
InvMin=-10
Penalty = -1

#CASE PARAMETERS
LT_s=14
LT_f=3
h=-2
b=-38
p=-0
C_f=-20
C_s=-10
cap_fast = 2
cap_slow = 2

initial_state = [3,4,2,4,2,4,2,4,2,4,2,4,2,4,2]
max_training_episodes = 10000000




# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def CreateActions(OrderFast,OrderSlow):
    Temp =[0 for z in range((OrderFast+1)*(OrderSlow+1))]
    z=0
    for i in itertools.product(list(range(0,OrderFast+1)),list(range(0,OrderSlow+1))):
        Temp[z]=i
        z+=1
    return Temp


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

            if depth_nn_hidden >= 1:
                self.hidden1 = slim.fully_connected(inputs=self.inputs, num_outputs=depth_nn_layers_hidden[0], activation_fn=activation_nn_hidden[0])
                self.state_out = slim.fully_connected(inputs=self.hidden1, num_outputs=depth_nn_out, activation_fn=activation_nn_out)
            if depth_nn_hidden >=2:
                self.hidden2=slim.fully_connected(inputs=self.hidden1,num_outputs=depth_nn_layers_hidden[1],activation_fn=activation_nn_hidden[1])
                self.state_out = slim.fully_connected(inputs=self.hidden2, num_outputs=depth_nn_out, activation_fn=activation_nn_out)
            if depth_nn_hidden >= 3:
                self.hidden3=slim.fully_connected(inputs=self.hidden2,num_outputs=depth_nn_layers_hidden[2],activation_fn=activation_nn_hidden[2])
                self.state_out = slim.fully_connected(inputs=self.hidden3, num_outputs=depth_nn_out, activation_fn=activation_nn_out)
            if depth_nn_hidden>= 4:
                self.hidden4=slim.fully_connected(inputs=self.hidden4,num_outputs=depth_nn_layers_hidden[3],activation_fn=activation_nn_hidden[3])
                self.state_out = slim.fully_connected(inputs=self.hidden4, num_outputs=depth_nn_out, activation_fn=activation_nn_out)



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
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * entropy_factor

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 1)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path,best_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.best_path = best_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(model_path + str(self.number)  + str(time.strftime(" %Y%m%d-%H%M%S")))
        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        self.bool_evaluating = None
        self.inv_vect = [3 for i in range(LT_s + 1)]
        self.best_solution = -9999999999999

    def Transition(self, Demand, Action):
        self.inv_vect[0] -= Demand
        self.inv_vect[0] += self.inv_vect[1]
        for i in range(1, LT_s):
            self.inv_vect[i] = self.inv_vect[i + 1]
        self.inv_vect[LT_s] = Action[1]
        self.inv_vect[LT_f] += Action[0]
        return self.inv_vect

    def Reward(self, s1, Action):
        reward = 0
        reward += np.ceil(float(Action[1]/cap_slow)) * C_s
        reward += np.ceil(float(Action[0]/cap_fast))* C_f
        if (s1[0] >= 0):
            reward += s1[0] * h
        else:
            reward += -s1[0] * b
        return reward / 1000000

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

    def work(self, max_episode_length, gamma, sess, coord, saver,saver_best):
        #best_solution = -99999999999
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while episode_count < max_training_episodes:# not coord.should_stop():
                if (episode_count % 50 == 0):
                    self.bool_evaluating = True
                else:
                    self.bool_evaluating = None

                sess.run(self.update_local_ops)
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

                while (d == False):  # modified to continue looping
                    # Take an action using probabilities from policy network output.
                    a_dist, v = sess.run([self.local_AC.policy, self.local_AC.value],
                                         feed_dict={self.local_AC.inputs: [s]})  # ,
                    a = np.random.choice(np.arange(len(a_dist[0])), p=a_dist[0])


                    if self.inv_vect[0] <= InvMin or self.inv_vect[0] >= InvMax:
                        d = True
                    else:
                        d = False


                    if d == False:
                        if self.bool_evaluating == True:
                            s1 = deepcopy(self.Transition(Demand[episode_step_count], actions[a]))
                        else:
                            s1 = deepcopy(self.Transition(int(np.round(np.random.uniform(0,5))), actions[
                                a]))
                    r = self.Reward(s1, actions[a])

                    if self.bool_evaluating == True:
                        eval_buffer.append([s, actions[a], r, s1, d, v[0, 0]])
                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    #print("[s,a,r,s1,d,v[0,0]]",[s,actions[a],r,s1,d,v[0,0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    s = deepcopy(s1)
                    total_steps += 1
                    episode_step_count += 1


                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) ==p_len_episode_buffer and d != True and episode_step_count != max_episode_length - 1 and self.bool_evaluating != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s]})[0, 0]  # ,

                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        # print(v_l,p_l,e_l,g_n,v_n)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                    if episode_step_count >= max_episode_length - 1:
                        break
                if self.bool_evaluating == True:
                    #print("EVALUATION", episode_reward / episode_step_count, episode_step_count)
                    if(episode_reward / episode_step_count > self.best_solution and episode_step_count==max_episode_length-1):
                        self.best_solution = episode_reward / episode_step_count
                        f= open(self.best_path + '/Train_'+str(self.number)+"/best_solution.txt","w")
                        f.write(str(self.best_solution) + ' ' + str (episode_step_count))
                        f.close()
                        saver_best.save(sess, self.best_path + '/Train_'+str(self.number) +'/model_' +' ' + str(episode_count) + '.cptk')


                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, Penalty)
                    # print("v_l",v_l,"p_l",p_l,"e_l",e_l,"g_n",g_n,"v_n",v_n)

                # Periodically save gifs of episodes, model parameters, and summary statistics.


                if episode_count % 250 == 0 and self.name == 'worker_0':
                    saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                    #print("Saved Model")

                mean_reward = np.mean(self.episode_rewards[-5:])
                mean_length = np.mean(self.episode_lengths[-5:])
                mean_value = np.mean(self.episode_mean_values[-5:])
                #print("mean reward per period: ", mean_reward / mean_length, "mean length:",
                #      mean_length, "mean value", mean_value)

                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward per period', simple_value=float(mean_reward/mean_length))
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
                episode_count += 1




def write_parameters(model_path,depth_nn_hidden,depth_nn_layers_hidden,depth_nn_out,entropy_factor,activation_nn_hidden,activation_nn_out,learning_rate,optimizer,activations,p_len_episode_buffer,max_episode_length,OrderFast,OrderSlow,LT_s,LT_f,InvMax,max_training_episodes):
    f = open(model_path + "/Parameters.txt","w")
    f.write("depth_nn_hidden: " + str(depth_nn_hidden))
    f.write("\ndepth_nn_layers_hidden "+str(depth_nn_layers_hidden))
    f.write("\ndepth_nn_out: " + str(depth_nn_out))
    f.write("\nentropy_factor "+str(entropy_factor))
    f.write("\nactivation_nn_hidden: " + str(activation_nn_hidden))
    f.write("\nactivation_nn_out "+str(activation_nn_out))
    f.write("\nLearning Rate: " + str(learning_rate))
    f.write("\noptimizer "+str(optimizer))
    f.write("\nactivations: " + str(activations))
    f.write("\np_len_episode_buffer " + str(p_len_episode_buffer))
    f.write("\nmax_episode_length: " + str(max_episode_length))
    f.write("\nOrderFast "+str(OrderFast))
    f.write("\nOrderSlow " + str(OrderSlow))
    f.write("\nLT_s " + str(LT_s))
    f.write("\nLT_f " + str(LT_f))
    f.write("\nh " + str(h))
    f.write("\nb " + str(b))
    f.write("\np " + str(p))
    f.write("\nC_f " + str(C_f))
    f.write("\nC_s " + str(C_s))
    f.write("\nInvMin " + str(InvMin))
    f.write("\nInvMax " + str(InvMax))
    f.write("\nPenalty " + str(Penalty))
    f.write("\ninitial_state " + str(initial_state))
    f.write("\nmax_training_episodes " + str(max_training_episodes))
    f.write("\nnb_workers"+str(nb_workers))
    f.write("\ncap_fast"+str(cap_fast))
    f.write("\ncap_slow"+str(cap_slow))
    f.close()
    return


actions=CreateActions(OrderFast,OrderSlow)#np.array([[0,0],[0,5],[5,0],[5,5]])
a_size = len(actions) # Agent can move Left, Right, or Fire



s_size = LT_s+1




tf.reset_default_graph()

load_model = False
model_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/model'
best_path = 'Logs/Logs_' + str(time.strftime("%Y%m%d-%H%M%S")) + '/best'

if not os.path.exists(model_path):
    os.makedirs(model_path)


with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = optimizer #tf.train.AdamOptimizer(learning_rate=learning_rate)
    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    num_workers = nb_workers#multiprocessing.cpu_count()  # Set workers to number of available CPU threads
    workers = []


    write_parameters(model_path,depth_nn_hidden,depth_nn_layers_hidden,depth_nn_out,entropy_factor,activation_nn_hidden,activation_nn_out,learning_rate,optimizer,activations,p_len_episode_buffer,max_episode_length,OrderFast,OrderSlow,LT_s,LT_f,InvMax,max_training_episodes)

    # Create worker classes
    for i in range(num_workers):
        if not os.path.exists(best_path + '/train_' + str(i)):
            os.makedirs(best_path + '/train_' + str(i))
        workers.append(Worker(None, i, s_size, a_size, trainer, model_path, best_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)
    saver_best = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.


        worker_threads = []
        temp_best_solutions = np.zeros([len(workers)])
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver,saver_best)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
        for index,worker in enumerate(workers):
            temp_best_solutions[index] = worker.best_solution
        best_solution_found = np.min(temp_best_solutions)

