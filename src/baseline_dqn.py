# Imports
import numpy as np
import tensorflow as tf
from schedules import LinearSchedule
from dfa import *
from game import *
import random, time, os.path, shutil
from network import get_MLP

"""
This baseline solves the problem using standard q-learning over the cross product 
between the LTL instruction and the MDP
"""

class BaselineDQN:
    def __init__(self, sess, num_actions, num_features, ltl, learning_params, feature_proxy):
        # Creating the network
        self.sess = sess
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_params = learning_params
        self.ltl_scope_name = "DQN_" + str(ltl).replace("&","AND").replace("|","OR").replace("!","NOT").replace("(","P1_").replace(")","_P2").replace("'","").replace(" ","").replace(",","_")
        self.create_network(learning_params.lr, learning_params.gamma)
        # Creating the experience replay buffer
        self.batch_size = learning_params.batch_size
        self.learning_starts = learning_params.learning_starts
        self.replay_buffer = DQNReplayBuffer(learning_params.buffer_size)
        # Adding the feature proxi (which include the ltl state to the feature vector)
        self.feature_proxy = feature_proxy # NOTE: we use this attribute at test time 


    def create_network(self, lr, gamma):
        total_features = self.num_features
        total_actions = self.num_actions

        # Inputs to the network
        self.s1 = tf.placeholder(tf.float64, [None, total_features])
        self.a = tf.placeholder(tf.int32)
        self.r = tf.placeholder(tf.float64)
        self.s2 = tf.placeholder(tf.float64, [None, total_features])
        self.done = tf.placeholder(tf.float64)

        # Creating target and current networks
        num_neurons = 64
        num_hidden_layers = 2
        
        with tf.variable_scope(self.ltl_scope_name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            q_values, q_target, self.update_target = get_MLP(self.s1, self.s2, total_features, total_actions, num_neurons, num_hidden_layers)
            
            # Q_values -> get optimal actions
            self.best_action = tf.argmax(q_values, 1)
            
            # Optimizing with respect to q_target
            action_mask = tf.one_hot(indices=self.a, depth=total_actions, dtype=tf.float64)
            q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)
            q_max = tf.reduce_max(q_target, axis=1)
            q_max = q_max * (1.0-self.done) # dead ends must have q_max equal to zero
            q_target_value = self.r + gamma * q_max
            q_target_value = tf.stop_gradient(q_target_value)
            loss = 0.5 * tf.reduce_sum(tf.square(q_current - q_target_value))
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train = optimizer.minimize(loss=loss)
            
            # Initializing the network values
            self.sess.run(tf.variables_initializer(self.get_network_variables()))
            self.update_target_network() #copying weights to target net

    def get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ltl_scope_name)

    def save_transition(self, s1, a, reward, s2, done):
        self.replay_buffer.add(s1, a.value, reward, s2, float(done))

    def get_steps(self):
        return len(self.replay_buffer)

    def learn(self):
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        s1, a, r, s2, done = self.replay_buffer.sample(self.batch_size)
        self.sess.run(self.train, {self.s1: s1, self.a: a, self.r: r, self.s2: s2, self.done: done})

    def get_best_action(self, s1):
        return self.sess.run(self.best_action, {self.s1: s1})

    def update_target_network(self):
        self.sess.run(self.update_target)

class DQNReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, s1, a, r, s2, done):
        data = (s1, a, r, s2, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        S1, A, R, S2, DONE = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            s1, a, r, s2, done = data
            S1.append(np.array(s1, copy=False))
            A.append(np.array(a, copy=False))
            R.append(r)
            S2.append(np.array(s2, copy=False))
            DONE.append(done)
        return np.array(S1), np.array(A), np.array(R), np.array(S2), np.array(DONE)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class FeatureProxy:
    def __init__(self, task):
        # NOTE: I have to add a representations for 'True' and 'False' 
        #       (even if they are not important in practice)
        num_states = len(task.dfa.ltl2state) - 2
        ltl2hotvector = {}
        for f in task.dfa.ltl2state:
            if f not in ['True', 'False']:
                aux = np.zeros((num_states), dtype=np.float64)
                aux[len(ltl2hotvector)] = 1.0
                ltl2hotvector[f] = aux
        ltl2hotvector["False"] = np.zeros((num_states), dtype=np.float64)
        ltl2hotvector["True"] = np.zeros((num_states), dtype=np.float64)
        self.ltl2hotvector = ltl2hotvector

    def get_features(self, task):
        s = task.get_features()
        ret = np.concatenate((s,self.ltl2hotvector[task.get_LTL_goal()])) # adding the DFA state to the features
        return ret

def _run_DQN(sess, policies, task_params, tester, curriculum, show_print):
    # Initializing parameters
    dqn = policies[task_params.ltl_task]
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    # Initializing parameters
    task = Game(task_params)
    actions = task.get_actions()
    feature_proxy = FeatureProxy(task)
    num_features = len(feature_proxy.get_features(task))

    # setting last parameters
    num_steps = learning_params.max_timesteps_per_task
    exploration = LinearSchedule(schedule_timesteps=int(learning_params.exploration_fraction * num_steps),initial_p=1.0,final_p=learning_params.exploration_final_eps)
    replay_buffer = DQNReplayBuffer(learning_params.buffer_size)
    training_reward = 0

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps, "actions...")
    for t in range(num_steps):
        # Getting the current state and ltl goal
        s1 = feature_proxy.get_features(task) # adding the DFA state to the features

        # Choosing an action to perform
        if random.random() < exploration.value(t): a = random.choice(actions)
        else: a = Actions(dqn.get_best_action(s1.reshape((1,num_features))))
        # updating the curriculum
        curriculum.add_step()
        
        # Executing the action
        reward = task.execute_action(a)
        training_reward += reward
        true_props = task.get_true_propositions()

        # Saving this transition
        s2 = feature_proxy.get_features(task) # adding the DFA state to the features
        done = task.ltl_game_over or task.env_game_over
        dqn.save_transition(s1, a, reward, s2, done)

        # Learning
        if dqn.get_steps() > learning_params.learning_starts and dqn.get_steps() % learning_params.train_freq == 0:
            dqn.learn()
            
        # Updating the target network
        if dqn.get_steps() > learning_params.learning_starts and dqn.get_steps() % learning_params.target_network_update_freq == 0:
            # Update target network periodically.
            dqn.update_target_network()

        # Printing
        if show_print and (dqn.get_steps()+1) % learning_params.print_freq == 0:
            print("Step:", dqn.get_steps()+1, "\tTotal reward:", training_reward, "\tSucc rate:", "%0.3f"%curriculum.get_succ_rate())

        # Testing
        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
            tester.run_test(curriculum.get_current_step(), sess, _test_DQN, policies)

        # Restarting the environment (Game Over)
        if done:
            # NOTE: Game over occurs for one of three reasons: 
            # 1) DFA reached a terminal state, 
            # 2) DFA reached a deadend, or 
            # 3) The agent reached an environment deadend (e.g. a PIT)
            task = Game(task_params) # Restarting

            # updating the hit rates
            curriculum.update_succ_rate(t, reward)
            if curriculum.stop_task(t):
                break
        
        # checking the steps time-out
        if curriculum.stop_learning():
            break

    if show_print: 
        print("Done! Total reward:", training_reward)


def _test_DQN(sess, task_params, learning_params, testing_params, policies):
    # Initializing parameters
    dqn = policies[task_params.ltl_task]
    feature_proxy = dqn.feature_proxy
    task = Game(task_params)

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Getting the current state and ltl goal
        s1 = feature_proxy.get_features(task)  # adding the DFA state to the features

        # Choosing an action to perform
        a = Actions(dqn.get_best_action(s1.reshape((1,len(s1)))))
        
        # Executing the action
        r_total += task.execute_action(a) * learning_params.gamma**t

        # Restarting the environment (Game Over)
        if task.ltl_game_over or task.env_game_over:
            break

    return r_total

def _initialize_policies(sess, learning_params, curriculum, tester):
    policies = {}
    for ltl_task in tester.get_LTL_tasks():
        task_aux = Game(tester.get_task_params(ltl_task))
        feature_proxy = FeatureProxy(task_aux)
        num_features = len(feature_proxy.get_features(task_aux))
        num_actions  = len(task_aux.get_actions())
        policies[ltl_task] = BaselineDQN(sess, num_actions, num_features, ltl_task, learning_params, feature_proxy)
    return policies
    
def run_experiments(alg_name, tester, curriculum, saver, num_times, show_print):

    # Running the tasks 'num_times'
    time_init = time.time()
    learning_params = tester.learning_params
    for t in range(num_times):
        # Setting the random seed to 't'
        random.seed(t)
        sess = tf.Session()

        # Reseting default values
        curriculum.restart()

        # Initializing policies
        policies = _initialize_policies(sess, learning_params, curriculum, tester)

        # Running the tasks
        while not curriculum.stop_learning():
            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            task = curriculum.get_next_task()
            task_params = tester.get_task_params(task)
            _run_DQN(sess, policies, task_params, tester, curriculum, show_print)
        tf.reset_default_graph()
        sess.close()

        # Backing up the results
        saver.save_results()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")
