import os
import random
import time
import numpy as np
import tensorflow as tf
from policy_bank import *
from schedules import LinearSchedule
from replay_buffer import ReplayBuffer
from dfa import *
from game import *
from test_utils import Loader, load_pkl


def run_experiments(tester, curriculum, saver, num_times, train_steps, show_print):
    time_init = time.time()
    tester_original = tester
    curriculum_original = curriculum
    loader = Loader(saver)
    train_dpath = os.path.join("../tmp", tester.experiment, "train_data")

    # Running the tasks 'num_times'
    for run_id in range(num_times):
        run_dpath = os.path.join(train_dpath, "run_%d" % run_id)
        # Overwrite 'tester' and 'curriculum' if incremental training
        tester_fpath = os.path.join(run_dpath, "tester.pkl")
        if os.path.exists(run_dpath) and os.path.exists(tester_fpath):
            tester = load_pkl(tester_fpath)
        else:
            tester = tester_original

        learning_params = tester.learning_params

        curriculum_fpath = os.path.join(run_dpath, "curriculum.pkl")
        if os.path.exists(run_dpath) and os.path.exists(curriculum_fpath):
            curriculum = load_pkl(curriculum_fpath)
            learning_params.learning_starts += curriculum.total_steps  # recollect 'replay_buffer'
            curriculum.incremental_learning(train_steps)
        else:
            curriculum = curriculum_original

        # Setting the random seed to 'run_id'
        random.seed(run_id)
        sess = tf.Session()

        # Reseting default values
        if not curriculum.incremental:
            curriculum.restart()

        # Initializing experience replay buffer
        replay_buffer = ReplayBuffer(learning_params.buffer_size)

        # Initializing policies per each subtask
        policy_bank = _initialize_policy_bank(sess, learning_params, curriculum, tester)
        # Load 'policy_bank' if incremental training
        policy_dpath = os.path.join(saver.policy_dpath, "run_%d" % run_id)
        if os.path.exists(policy_dpath) and os.listdir(policy_dpath):
            loader.load_policy_bank(run_id, sess)
        if show_print:
            print("Policy bank initialization took: %0.2f mins" % ((time.time() - time_init)/60))

        # Running the tasks
        num_tasks = 0
        while not curriculum.stop_learning():
            task = curriculum.get_next_task()
            if show_print:
                print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
                print("%d Current task: %d, %s" % (num_tasks, curriculum.current_task, str(task)))
            task_params = tester.get_task_params(task)
            _run_LPOPL(sess, policy_bank, task_params, tester, curriculum, replay_buffer, show_print)
            num_tasks += 1
        # Save 'policy_bank' for incremental training and transferring
        saver.save_policy_bank(policy_bank, run_id)
        # Backing up the results
        saver.save_results()
        # Save 'tester' and 'curriculum' for incremental training
        saver.save_train_data(curriculum, run_id)

        tf.reset_default_graph()
        sess.close()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f" % ((time.time() - time_init)/60), "mins")


def _initialize_policy_bank(sess, learning_params, curriculum, tester, load_tf = True):
    task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
    num_actions = len(task_aux.get_actions())
    num_features = task_aux.get_num_features()
    start_time = time.time()
    policy_bank = PolicyBank(sess, num_actions, num_features, learning_params)
    print("policy bank initialization took %0.2f mins" % ((time.time() - start_time)/60))
    for idx, f_task in enumerate(tester.get_LTL_tasks()):
        start_time = time.time()
        dfa = DFA(f_task)
        print("%d processing LTL: %s" % (idx, str(f_task)))
        print("took %0.2f mins to construct DFA" % ((time.time() - start_time)/60))
        start_time = time.time()
        for ltl in dfa.ltl2state:
            # this method already checks that the policy is not in the bank and it is not 'True' or 'False'
            policy_bank.add_LTL_policy(ltl, f_task, dfa, load_tf = load_tf)
        print("took %0.2f mins to add policy" % ((time.time() - start_time)/60))
    policy_bank.reconnect()  # -> creating the connections between the neural nets
    print("\n", policy_bank.get_number_LTL_policies(), "sub-tasks were extracted!\n")
    return policy_bank


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
