import random
import time
import numpy as np
import tensorflow as tf
from policy_bank import *
from schedules import LinearSchedule
from replay_buffer import ReplayBuffer
from dfa import *
from game import *


def _run_LPOPL(sess, policy_bank, task_params, tester, curriculum, replay_buffer, show_print):
    # Initializing parameters
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    # Initializing the game
    task = Game(task_params)
    actions = task.get_actions()

    # Initializing parameters
    num_features = task.get_num_features()
    num_steps = learning_params.max_timesteps_per_task
    exploration = LinearSchedule(schedule_timesteps=int(learning_params.exploration_fraction * num_steps), initial_p=1.0, final_p=learning_params.exploration_final_eps)
    training_reward = 0

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps, "actions...")
    for t in range(num_steps):
        # Getting the current state and ltl goal
        ltl_goal = task.get_LTL_goal()
        s1 = task.get_features()

        # Choosing an action to perform
        if random.random() < exploration.value(t): a = random.choice(actions)
        else: a = Actions(policy_bank.get_best_action(ltl_goal, s1.reshape((1, num_features))))
        # updating the curriculum
        curriculum.add_step()

        # Executing the action
        reward = task.execute_action(a)
        training_reward += reward
        true_props = task.get_true_propositions()

        # Saving this transition
        s2 = task.get_features()
        next_goals = np.zeros((policy_bank.get_number_LTL_policies(),), dtype=np.float64)
        for ltl in policy_bank.get_LTL_policies():
            ltl_id = policy_bank.get_id(ltl)
            if task.env_game_over:
                ltl_next_id = policy_bank.get_id("False")  # env deadends are equal to achive the 'False' formula
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
            tester.run_test(curriculum.get_current_step(), sess, _test_LPOPL, policy_bank, num_features)

        # Restarting the environment (Game Over)
        if task.ltl_game_over or task.env_game_over:
            # NOTE: Game over occurs for one of three reasons:
            # 1) DFA reached a terminal state,
            # 2) DFA reached a deadend, or
            # 3) The agent reached an environment deadend (e.g. a PIT)
            task = Game(task_params)  # Restarting

            # updating the hit rates
            curriculum.update_succ_rate(t, reward)
            if curriculum.stop_task(t):
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

    if show_print:
        print("Done! Total reward:", training_reward)


def _test_LPOPL(sess, task_params, learning_params, testing_params, policy_bank, num_features):
    # Initializing parameters
    task = Game(task_params)

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Getting the current state and ltl goal
        s1 = task.get_features()

        # Choosing an action to perform
        a = Actions(policy_bank.get_best_action(task.get_LTL_goal(), s1.reshape((1, num_features))))

        # Executing the action
        r_total += task.execute_action(a) * learning_params.gamma**t

        # Restarting the environment (Game Over)
        if task.ltl_game_over or task.env_game_over:
            break
    return r_total


def _initialize_policy_bank_v1(sess, learning_params, curriculum, tester):
    task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
    num_actions  = len(task_aux.get_actions())
    num_features = task_aux.get_num_features()
    policy_bank = PolicyBank(sess, num_actions, num_features, learning_params)

    time_dfa_construction = 0 # Time taken to compile component DFAs
    time_policy_init = 0 # Time taken to initialize policy bank

    for (i,f_task) in enumerate(tester.get_LTL_tasks()):
        start = time.time()
        dfa = DFA(f_task)
        stop = time.time()
        time_dfa_construction = time_dfa_construction + (stop - start)
        print(f'Formula {i}, DFA construction time: {stop - start}')

        start = time.time()

        print(f'Formula{i}, number of DFA states: {len(dfa.ltl2state)}')

        for ltl in dfa.ltl2state:
            # this method already checks that the policy is not in the bank and it is not 'True' or 'False'
            policy_bank.add_LTL_policy(ltl, f_task, dfa)
        stop = time.time()
        time_policy_init = time_policy_init + (stop - start)
        print(f'Formula {i}: Policy initialization time: {stop - start}')

    policy_bank.reconnect()  # -> creating the connections between the neural nets

    # print("\n", policy_bank.get_number_LTL_policies(), "sub-tasks were extracted!\n")
    return policy_bank

def _initialize_policy_bank(sess, learning_params, curriculum, tester):
    task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
    num_actions  = len(task_aux.get_actions())
    num_features = task_aux.get_num_features()
    policy_bank = PolicyBank(sess, num_actions, num_features, learning_params)

    time_dfa_construction = 0 # Time taken to compile component DFAs
    time_policy_init = 0 # Time taken to initialize policy bank

    #Separating LTL set construction and policy initialization
    ltl_set = set()
    for (i,f_task) in enumerate(tester.get_LTL_tasks()):
        start = time.time()
        dfa = DFA(f_task)
        ltl_set = ltl_set | set(dfa.ltl2state.keys())
        stop = time.time()
        time_dfa_construction = time_dfa_construction + (stop - start)
        print(f'Formula {i}, DFA construction time: {stop - start}')
        start = time.time()
    print(f'LTL set contains {len(ltl_set)} formulas')
    print(f'Total DFA construction time: {time_dfa_construction}')

    for (i,ltl) in enumerate(ltl_set):
        #print(f'Formula{i} of {len(ltl_set)}')
        start = time.time()
        policy_bank.add_LTL_policy(ltl, f_task, dfa)
        stop = time.time()
        time_policy_init = time_policy_init + (stop - start)
        print(f'Formula {i} of {len(ltl_set)}: Policy initialization time: {stop - start}')
    print(f'Total policy initialization time {time_policy_init}')

    policy_bank.reconnect()  # -> creating the connections between the neural nets

    # print("\n", policy_bank.get_number_LTL_policies(), "sub-tasks were extracted!\n")
    return policy_bank


def run_experiments(tester, curriculum, saver, num_times, show_print):
    # Running the tasks 'num_times'
    time_init = time.time()
    learning_params = tester.learning_params
    for t in range(num_times):
        # Setting the random seed to 't'
        random.seed(t)
        sess = tf.Session()

        # Reseting default values
        curriculum.restart()

        # Initializing experience replay buffer
        replay_buffer = ReplayBuffer(learning_params.buffer_size)

        # Initializing policies per each subtask
        policy_bank = _initialize_policy_bank(sess, learning_params, curriculum, tester)
        if show_print:
            print("Policy bank initialization took: %0.2f mins" % ((time.time() - time_init)/60))

        # Running the tasks
        while not curriculum.stop_learning():
            task = curriculum.get_next_task()
            if show_print:
                print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
                print("Current task: ", task)
            task_params = tester.get_task_params(task)
            _run_LPOPL(sess, policy_bank, task_params, tester, curriculum, replay_buffer, show_print)
        saver.save_policy_bank(policy_bank, t)
        # Backing up the results
        saver.save_results()

        tf.reset_default_graph()
        sess.close()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f" % ((time.time() - time_init)/60), "mins")
