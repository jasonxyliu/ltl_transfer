# Imports
import numpy as np
import tensorflow as tf
from schedules import LinearSchedule
from dfa import *
from game import *
import random, time, os, shutil
from policy_bank import PolicyBank
from replay_buffer import ReplayBuffer
from test_utils import Saver
from tasks import get_option, get_option_night
from ltl_progression import extract_propositions

class MetaController:
    def __init__(self, subpolicies, gamma, use_dfa, alpha=0.7, epsilon=0.1):
        self.Q = {}
        self.subpolicies = list(subpolicies)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_dfa = use_dfa # 'use_dfa = True' means that we use the DFA to prune useless options from consideration
    
    def learn(self, s1, a, r, s2, done, steps, dfa, ltl):
        if done:
            q_new = r
        else:
            if r > 0: print("ERROR!")
            q_new = (r + (self.gamma**steps)*self._get_max_q_value(s2, dfa, ltl))

        self.Q[s1][a] += self.alpha*(q_new - self.Q[s1][a]) 

    def _get_options(self, s, dfa, ltl):
        # removing the option of picking a subgoal that is already true
        true_props = s[2]
        ret = [p for p in self.subpolicies if p not in true_props]
        if self.use_dfa:
            ret = [p for p in ret if ltl not in [dfa.progress_LTL(ltl, p), "False"]]   
        return ret

    def get_action_epsilon_greedy(self, s, dfa, ltl):
        if random.random() < self.epsilon:
            return random.choice(self._get_options(s, dfa, ltl))
        return self.get_best_action(s, dfa, ltl)

    def get_best_action(self, s, dfa, ltl):
        true_props = s[2]
        # I have to return an subgoal that it is not already true...
        l = [(self._get_q_value(s,a),a) for a in self._get_options(s, dfa, ltl)]
        random.shuffle(l)
        l = sorted(l, key=lambda x: x[0], reverse=True)
        return l[0][1]

    def _get_max_q_value(self, s, dfa, ltl):
        return max([self._get_q_value(s,a) for a in self._get_options(s, dfa, ltl)])

    def _get_q_value(self, s, a):
        if s not in self.Q:
            self.Q[s] = {}
        if a not in self.Q[s]:
            self.Q[s][a] = 1
        return self.Q[s][a]


def _get_features_meta_controller(task):
    return task.agent.i, task.agent.j, task.get_true_propositions(), task.get_LTL_goal()

def _get_LTL_formula(task, subgoal):
    if task.consider_night:
        return get_option_night(subgoal)
    return get_option(subgoal)


def _run_HRL(sess, meta_controllers, policy_bank, task_params, tester, curriculum, replay_buffer, show_print):
    """
    Strategy:
        - I'll learn a tabular metacontroller over the posible subpolicies
        - Initialice a regular policy bank with eventually subpolicies (e.g. eventually(a), eventually(b), ...)
        - Learn as usual
    """

    # Initializing parameters
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    # Initializing parameters
    task = Game(task_params)
    actions = task.get_actions()
    
    # Creating the neuralnet
    num_features = len(task.get_features())

    # Creating the meta-controller
    meta_controller = meta_controllers[task_params.ltl_task]

    # Initializing parameters
    num_steps = learning_params.max_timesteps_per_task
    exploration = LinearSchedule(schedule_timesteps=int(learning_params.exploration_fraction * num_steps),initial_p=1.0,final_p=learning_params.exploration_final_eps)
    training_reward = 0

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps, "actions...")
    t = 0
    curriculum_stop = False
    while t < learning_params.max_timesteps_per_task and not curriculum_stop:
        # selecting a macro action from the meta controller
        mc_s1 = _get_features_meta_controller(task)
        mc_r  = []
        mc_a  = meta_controller.get_action_epsilon_greedy(mc_s1, task.dfa, task.get_LTL_goal())
        mc_done = False

        while mc_a not in task.get_true_propositions():
            # Getting the current state and ltl goal
            s1 = task.get_features()

            # Choosing an action according to option mc_a
            if random.random() < exploration.value(t): 
                a = random.choice(actions)
            else: 
                a = Actions(policy_bank.get_best_action(_get_LTL_formula(task, mc_a), s1.reshape((1,num_features))))
            
            # updating the curriculum
            curriculum.add_step()
            
            # Executing the action
            reward = task.execute_action(a)
            training_reward += reward
            true_props = task.get_true_propositions()

            # updating the reward for the meta controller
            mc_r.append(reward)

            # Saving this transition 
            s2 = task.get_features()
            next_goals = np.zeros((policy_bank.get_number_LTL_policies(),), dtype=np.float64)
            for ltl in policy_bank.get_LTL_policies():
                ltl_id = policy_bank.get_id(ltl)
                if task.env_game_over:
                    ltl_next_id = policy_bank.get_id("False") # env deadends are equal to achive the 'False' formula
                else: 
                    ltl_next_id = policy_bank.get_id(policy_bank.get_policy_next_LTL(ltl, true_props))
                next_goals[ltl_id-2] = ltl_next_id
            replay_buffer.add(s1, a.value, s2, next_goals)

            # Learning
            if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                S1, A, S2, Goal = replay_buffer.sample(learning_params.batch_size)
                policy_bank.learn(S1, A, S2, Goal)
                
            # Updating the target network
            if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.target_network_update_freq == 0:
                # Update target network periodically.
                policy_bank.update_target_network()

            # Printing
            if show_print and (curriculum.get_current_step()+1) % learning_params.print_freq == 0:
                print("Step:", curriculum.get_current_step()+1, "\tTotal reward:", training_reward, "\tSucc rate:", "%0.3f"%curriculum.get_succ_rate())

            # Testing
            if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
                tester.run_test(curriculum.get_current_step(), sess, _test_HRL, meta_controllers, policy_bank, num_features)

            # Restarting the environment (Game Over)
            if task.ltl_game_over or task.env_game_over:
                task = Game(task_params) # Restarting
                mc_done = True

                # updating the hit rates
                curriculum.update_succ_rate(t, reward)
                if curriculum.stop_task(t):
                    curriculum_stop = True
            
            # checking the steps time-out
            if curriculum.stop_learning():
                curriculum_stop = True

            t += 1
            if t == learning_params.max_timesteps_per_task or curriculum_stop or mc_done: 
                break
        
        # learning on the meta controller
        mc_s2 = _get_features_meta_controller(task)
        meta_controller.learn(mc_s1, mc_a, _get_discounted_reward(mc_r, learning_params.gamma), mc_s2, mc_done, len(mc_r), task.dfa, task.get_LTL_goal())

    if show_print: 
        print("Done! Total reward:", training_reward)


def _get_discounted_reward(r_all, gamma):
    dictounted_r = 0
    for r in r_all[::-1]:
        dictounted_r = r + gamma*dictounted_r
    return dictounted_r

def _test_HRL(sess, task_params, learning_params, testing_params, meta_controllers, policy_bank, num_features):
    # Initializing parameters
    task = Game(task_params)
    meta_controller = meta_controllers[task_params.ltl_task]

    # Starting interaction with the environment
    r_total = 0
    t = 0
    while t < testing_params.num_steps:
        # selecting a macro action from the meta controller
        mc_s1 = _get_features_meta_controller(task)
        mc_a  = meta_controller.get_best_action(mc_s1, task.dfa, task.get_LTL_goal())

        while mc_a not in task.get_true_propositions():
            # Getting the current state and ltl goal
            s1 = task.get_features()

            # Choosing an action to perform
            a = Actions(policy_bank.get_best_action(_get_LTL_formula(task, mc_a), s1.reshape((1,num_features))))
            
            # Executing the action
            r_total += task.execute_action(a) * learning_params.gamma**t

            t += 1
            # Restarting the environment (Game Over)
            if task.ltl_game_over or task.env_game_over or t == testing_params.num_steps:
                return r_total

def _get_sub_goals(ltl_task):
    vocabulary = extract_propositions(ltl_task)
    return "".join([v for v in vocabulary if v not in "ns"]) # We don't consider 'n' or 's' as valid subgoals

def _initialize_option_policies(sess, subgoals, learning_params, curriculum, tester):
    task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
    num_features = len(task_aux.get_features())
    num_actions  = len(task_aux.get_actions())

    policy_bank = PolicyBank(sess, num_actions, num_features, learning_params)
    for s in subgoals:
        ltl = _get_LTL_formula(task_aux, s)
        dfa = DFA(ltl)
        policy_bank.add_LTL_policy(ltl, dfa)
    policy_bank.reconnect() # -> creating the connections between the neural nets

    print("\n", policy_bank.get_number_LTL_policies(), "options were extracted for these tasks!\n")
    return policy_bank


def run_experiments(alg_name, tester, curriculum, saver, num_times, show_print, use_dfa):
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
    time_init = time.time()
    for t in range(num_times):
        # Setting the random seed to 't'
        random.seed(t)
        sess = tf.Session()

        # Reseting default values
        curriculum.restart()
        
        # Initializing experience replay buffer
        replay_buffer = ReplayBuffer(learning_params.buffer_size)
        # initializing the meta controllers
        meta_controllers = {}
        subgoals = set("".join([_get_sub_goals(ta) for ta in tester.tasks])) # the standard version shouldn't take advantage of the LTL formula
        for ta in tester.tasks:
            meta_controllers[ta] = MetaController(subgoals, learning_params.gamma, use_dfa)
        # initializing option's policies
        policy_bank = _initialize_option_policies(sess, subgoals, learning_params, curriculum, tester)

        # Running the tasks
        while not curriculum.stop_learning():
            if show_print: print("Current step:", curriculum.get_current_step())
            task = curriculum.get_next_task()
            task_params = tester.get_task_params(task)
            _run_HRL(sess, meta_controllers, policy_bank, task_params, tester, curriculum, replay_buffer, show_print)

        tf.reset_default_graph()
        sess.close()

        # Backing up the results
        saver.save_results()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")
