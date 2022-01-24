import os
import random, time, shutil
import dill
from multiprocessing import pool
from collections import defaultdict
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
    num_features = len(task.get_features())
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


def _initialize_policy_bank(sess, learning_params, curriculum, tester):
    task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
    num_features = len(task_aux.get_features())
    num_actions  = len(task_aux.get_actions())
    policy_bank = PolicyBank(sess, num_actions, num_features, learning_params)
    for f_task in tester.get_LTL_tasks():
        dfa = DFA(f_task)
        for ltl in dfa.ltl2state:
            # this method already checks that the policy is not in the bank and it is not 'True' or 'False'
            policy_bank.add_LTL_policy(ltl, f_task, dfa)
    policy_bank.reconnect()  # -> creating the connections between the neural nets

    print("\n", policy_bank.get_number_LTL_policies(), "sub-tasks were extracted!\n")
    return policy_bank


def run_experiments(tester, curriculum, saver, loader, num_times, load_trained, show_print):
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

        if load_trained:
            loader.load_policy_bank(t, sess)
            task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
            num_features = len(task_aux.get_features())
            tester.run_test(-1, sess, _test_LPOPL, policy_bank, num_features)  # -1 to signal test after restore models
            # print(tester.results)
        else:
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
            saver.save_transfer_results()

        # Relabel state-centric options to transition-centric options
        relabel(tester, curriculum, policy_bank)

        # saver.save_classifier_data(policy_bank, curriculum, t)
        # run_rollouts(tester, policy_bank)
        # load_classifier_results(tester, policy_bank, curriculum)
        # run_transfer_experiments(tester, policy_bank)

        tf.reset_default_graph()
        sess.close()

    # Showing results
    tester.show_results()
    tester.show_transfer_results()
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")


def run_rollouts(tester, policy_bank):
    # run single rollout in parallel
    for ltl in policy_bank.get_LTL_policies():
        print(ltl)
        policy = policy_bank.policies[policy_bank.get_id(ltl)]
        print(policy.get_edge_labels())


def load_classifier_results(tester, policy_bank):
    """
    Aggregate results from learning classifiers in parallel
    """
    results_fpath = os.path.join("results", "classifier", tester.map_id, "results.txt")

    with open(os.path.join("results", "classifier", "states.pkl"), "rb") as file:
        id2state = dill.load(file)

    with open(results_fpath, "r") as file:
        lines = file.readlines()

    policy2edge2locs = defaultdict(lambda: defaultdict(list))
    for line in lines:
        if line:
            policy_id, state_id, edge = line.strip().split(" ")
            policy = policy_bank.policies[int(policy_id)]
            loc = id2state[int(state_id)]
            policy2edge2locs[policy][edge].append(loc)

    for policy, edge2locs in policy2edge2locs.items():
        for edge, classifier in edge2locs.items():
            policy.add_initiation_set_classifier(edge, classifier)


def relabel(tester, curriculum, policy_bank):
    """
    Rollout every state-centric option's policy to try to satisfy each outgoing edge
    to learn an initiation set classifier for each relabeled transition-centric option
    """
    for ltl_idx, ltl in enumerate(policy_bank.get_LTL_policies()):
        # if policy_bank.get_id(ltl) != 3:
        #     continue
        print(ltl_idx, ": ltl (sub)task: ", ltl)
        policy = policy_bank.policies[policy_bank.get_id(ltl)]
        print("edges: ", policy.get_edge_labels())
        learn_naive_classifier(tester, curriculum, policy_bank, ltl, max_depth=curriculum.num_steps)
        print("\n")


def learn_naive_classifier(tester, curriculum, policy_bank, ltl, n_rollouts=100, max_depth=100):
    """
    After n_rollouts from a loc, this loc is in the initiation set of the option
    whose policy satisfies the edge subtask more times than other option's policies
    The initiation sets of all options are non-overlapping.
    """
    task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
    edge2locs = defaultdict(list)  # classifier for every edge

    rollout(tester, policy_bank, ltl, (10, 10), n_rollouts, max_depth)

    # for y in range(task_aux.map_height):
    #     for x in range(task_aux.map_width):
    #         if task_aux.is_valid_agent_loc(x, y):
    #             edge2locs[rollout(tester, policy_bank, ltl, (x, y), n_rollouts, max_depth)].append((x, y))
    #
    # print(edge2locs)
    #
    # policy = policy_bank.policies[policy_bank.get_id(ltl)]
    # for edge, classifier in edge2locs.items():
    #     policy.add_initiation_set_classifier(edge, classifier)


def rollout(tester, policy_bank, ltl, init_loc, n_rollouts, max_depth):
    """
    Rollout trained policy from init_loc to see which outgoing edge it satisfies
    """
    print("init_loc: ", init_loc)
    edge2hits = defaultdict(int)
    for rollout in range(n_rollouts):
        # print("rollout:", rollout)

        task = Game(tester.get_task_params(policy_bank.policies[policy_bank.get_id(ltl)].f_task, ltl, init_loc))
        # print(task.dfa.state)
        # print(policy_bank.policies[policy_bank.get_id(ltl)].f_task)

        depth = 0
        traversed_edge = None
        while not task.ltl_game_over and not task.env_game_over and depth <= max_depth:
            s1 = task.get_features()
            action = Actions(policy_bank.get_best_action(ltl, s1.reshape((1, len(task.get_features())))))
            prev_state = task.dfa.state
            _ = task.execute_action(action)
            # print(prev_state, action, task.dfa.state)
            if prev_state != task.dfa.state:
                traversed_edge = task.dfa.nodelist[prev_state][task.dfa.state]
                # print(traversed_edge)
                break
            depth += 1
        if traversed_edge:
            edge2hits[traversed_edge] += 1
    print(edge2hits)
    max_edge = None
    if edge2hits:
        max_edge = max(edge2hits.items(), key=lambda kv: kv[1])[0]
    return max_edge


def run_transfer_experiments(tester, policy_bank):
    transfer_tasks = tester.get_transfer_tasks()
    training_edges = set([policy.get_edge_labels() for policy in policy_bank.policies])

    for transfer_task in transfer_tasks:
        new_dfa = DFA(transfer_task)
        # wrapper for NetworkX

        # graph search through DFA
        # check if an edge seen in training
        # find a path with all seen edges
