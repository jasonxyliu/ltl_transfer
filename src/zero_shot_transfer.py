import os
import random, time, shutil
import json
import dill
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
from lpopl import _initialize_policy_bank, _test_LPOPL
from policy_bank import *
from dfa import *
from game import *
from run_single_worker import single_worker_rollouts


def run_experiments(tester, curriculum, saver, loader, run_id):
    # Running the tasks 'num_times'
    time_init = time.time()
    learning_params = tester.learning_params

    random.seed(run_id)
    sess = tf.Session()

    # Reseting default values
    curriculum.restart()

    # Initializing policies per each subtask
    policy_bank = _initialize_policy_bank(sess, learning_params, curriculum, tester)

    # print("loading policy bank in lpopl")
    loader.load_policy_bank(run_id, sess)
    # print("policy_dpath in lpopl: ", loader.saver.policy_dpath)
    task_aux = Game(tester.get_task_params(tester.get_LTL_tasks()[0]))
    num_features = task_aux.get_num_features()
    tester.run_test(-1, sess, _test_LPOPL, policy_bank, num_features)  # -1 to signal test after restore models
    # print(tester.results)

    # Relabel state-centric options to transition-centric options
    # relabel(tester, saver, curriculum, policy_bank)
    relabel_parallel(tester, saver, curriculum, run_id, policy_bank)
    policy2edge2loc2prob = construct_initiation_set_classifiers(saver)
    task2sol = zero_shot_transfer(tester, policy_bank, policy2edge2loc2prob)
    # saver.save_transfer_results()

    tf.reset_default_graph()
    sess.close()

    # Showing transfer results
    tester.show_transfer_results()
    print("Time:", "%0.2f" % ((time.time() - time_init)/60), "mins")


def relabel_parallel(tester, saver, curriculum, run_id, policy_bank, n_rollouts=100):
    """
    A worker runs n_rollouts from a specific location for a specific LTL
    """
    task_aux = Game(tester.get_task_params(tester.get_LTL_tasks()[0]))
    state2id = saver.save_training_data(task_aux)
    worker_commands = []
    for ltl_idx, ltl in enumerate(policy_bank.get_LTL_policies()):
        ltl_id = policy_bank.get_id(ltl)
        # if ltl_id not in [12, 16, 30]:
        #     continue
        # print("index ", ltl_idx, ". ltl (sub)task: ", ltl, ltl_id)

        # x_tests = np.random.randint(1, 20, size=1)
        # y_tests = np.random.randint(1, 20, size=1)
        # test_locs = list(zip(x_tests, y_tests))
        # test_locs = [(5, 15), (10, 10)]
        # print("test_locs: ", test_locs)
        for x in range(task_aux.map_width):
            for y in range(task_aux.map_height):
                # if (x, y) not in test_locs:
                #     continue
                if task_aux.is_valid_agent_loc(x, y):
                    # create directory to store results from a single worker
                    # saver.create_worker_directory(ltl_id, state2id[(x, y)])
                    # create command to run a single worker
                    args = "--algo=%s --tasks_id=%d --map_id=%d --run_id=%d --ltl_id=%d --state_id=%d --n_rollouts=%d --max_depth=%d" % (
                        saver.alg_name, tester.tasks_id, tester.map_id, run_id, ltl_id, state2id[(x, y)], n_rollouts, curriculum.num_steps)
                    worker_commands.append("python3 run_single_worker.py %s" % args)

    with Pool(processes=len(worker_commands)) as pool:
        retvals = pool.map(os.system, worker_commands)
    for retval, worker_command in zip(retvals, worker_commands):
        if retval:  # os.system exit code: 0 means correct execution
            print("Command failed: ", retval, worker_command)
            retval = os.system(worker_command)

    aggregate_rollout_results(task_aux, saver, policy_bank, n_rollouts)


def aggregate_rollout_results(task_aux, saver, policy_bank, n_rollouts):
    """
    Aggregate results saved locally by parallel workers for learning classifiers
    """
    policy2loc2edge2hits_json = {"n_rollouts": n_rollouts}
    policy2loc2edge2hits_pkl = {"n_rollouts": n_rollouts}
    id2ltl = {}
    for ltl_idx, ltl in enumerate(policy_bank.get_LTL_policies()):
        ltl_id = policy_bank.get_id(ltl)
        id2ltl[ltl_id] = ltl
        # if ltl_id not in [12, 16, 30]:
        #     continue
        policy2loc2edge2hits_json[str(ltl)] = {}
        policy2loc2edge2hits_pkl[ltl] = {}
        for x in range(task_aux.map_width):
            for y in range(task_aux.map_height):
                # if (x, y) not in [(5, 15), (10, 10)]:
                #     continue
                if task_aux.is_valid_agent_loc(x, y):
                    worker_fpath = os.path.join(saver.classifier_dpath, "ltl%d_state%d-%d_" % (ltl_id, x, y))
                    with open(worker_fpath+"rollout_results_parallel.pkl", "rb") as file:
                        rollout_results = dill.load(file)
                    # try:  # for local testing parallel rollout with a few random locs
                    #     with open(worker_fpath + "rollout_results_parallel.pkl", "rb") as file:
                    #         rollout_results = dill.load(file)
                    # except IOError:
                    #     continue
                    policy2loc2edge2hits_json[str(ltl)][str((x, y))] = rollout_results["edge2hits"]
                    policy2loc2edge2hits_pkl[ltl][(x, y)] = rollout_results["edge2hits"]
    policy2loc2edge2hits_json["ltls"] = id2ltl
    policy2loc2edge2hits_pkl["ltls"] = id2ltl
    saver.save_rollout_results("rollout_results_parallel", policy2loc2edge2hits_json, policy2loc2edge2hits_pkl)


def relabel(tester, saver, curriculum, policy_bank, n_rollouts=100):
    """
    To construct a relabeled transition-centric option,
    rollout every state-centric option's policy to try to satisfy each outgoing edge
    to learn an initiation set classifier for that edge
    """
    policy2loc2edge2hits = {"n_rollouts": n_rollouts}
    id2ltl = {}
    for ltl_idx, ltl in enumerate(policy_bank.get_LTL_policies()):
        ltl_id = policy_bank.get_id(ltl)
        id2ltl[ltl_id] = ltl
        if ltl_id != 9:
            continue
        print(ltl_idx, ". ltl (sub)task: ", ltl, ltl_id)
        policy = policy_bank.policies[policy_bank.get_id(ltl)]
        print("outgoing edges: ", policy.get_edge_labels())
        loc2edge2hits = learn_naive_classifier(tester, policy_bank, ltl_id, n_rollouts, curriculum.num_steps)
        policy2loc2edge2hits[str(ltl)] = loc2edge2hits
        print("\n")
    print(policy2loc2edge2hits)
    policy2loc2edge2hits["ltls"] = id2ltl
    saver.save_rollout_results("rollout_results", policy2loc2edge2hits)


def learn_naive_classifier(tester, policy_bank, ltl, n_rollouts=100, max_depth=100):
    """
    After n_rollouts from every loc, the initiation sets of an edge-centric option is
    the probability of success from every loc
    LPOPL learns deterministic policies, so at most 1 edge transition would occur from a loc
    """
    # edge2hits = rollout(tester, policy_bank, ltl, (13, 10), n_rollouts, max_depth)
    # loc2edge2hits = {"(13, 10)": edge2hits}

    loc2edge2hits = {}
    # edge2locs = defaultdict(list)  # classifier for every edge
    task_aux = Game(tester.get_task_params(ltl))
    for x in range(task_aux.map_width):
        for y in range(task_aux.map_height):
            # if (x, y) != (10, 10):
            #     continue
            if task_aux.is_valid_agent_loc(x, y):
                print("init_loc: ", (x, y))
                edge2hits = rollout(tester, policy_bank, ltl, (x, y), n_rollouts, max_depth)
                loc2edge2hits[str((x, y))] = edge2hits
                # max_edge = None
                # if edge2hits:
                #     max_edge = max(edge2hits.items(), key=lambda kv: kv[1])[0]
                # edge2locs[max_edge].append((x, y))
    # print(edge2locs)

    # process rollout results to construct a classifier
    edge2loc2prob = defaultdict(dict)
    for loc, edge2hits in loc2edge2hits.items():
        for edge, hits in edge2hits.items():
            edge2loc2prob[edge][loc] = hits / n_rollouts

    policy = policy_bank.policies[policy_bank.get_id(ltl)]
    for edge, classifier in edge2loc2prob.items():
        policy.add_initiation_set_classifier(edge, classifier)

    return loc2edge2hits


def rollout(tester, policy_bank, ltl, init_loc, n_rollouts, max_depth):
    """
    Rollout trained policy from init_loc to see which outgoing edges it satisfies
    """
    edge2hits = defaultdict(int)
    task_aux = Game(tester.get_task_params(policy_bank.policies[policy_bank.get_id(ltl)].f_task, ltl))
    initial_state = task_aux.dfa.state  # get DFA initial state before progressing on agent init_loc
    for rollout in range(n_rollouts):
        # print("init_loc: ", init_loc)
        # print("initial_state: ", initial_state)
        # print("rollout:", rollout)

        task = Game(tester.get_task_params(policy_bank.policies[policy_bank.get_id(ltl)].f_task, ltl, init_loc))
        # print("cur_state: ", task.dfa.state)
        # print("full ltl: ", policy_bank.policies[policy_bank.get_id(ltl)].f_task)

        traversed_edge = None
        if initial_state != task.dfa.state:  # if agent starts at a given loc that triggers a desired transition
            traversed_edge = task.dfa.nodelist[initial_state][task.dfa.state]
            # print("before while: ", traversed_edge)
        depth = 0
        while not traversed_edge and not task.ltl_game_over and not task.env_game_over and depth <= max_depth:
            s1 = task.get_features()
            action = Actions(policy_bank.get_best_action(ltl, s1.reshape((1, len(task.get_features())))))
            prev_state = task.dfa.state
            _ = task.execute_action(action)
            # print(prev_state, action, task.dfa.state)
            if prev_state != task.dfa.state:
                traversed_edge = task.dfa.nodelist[prev_state][task.dfa.state]
                # print("in while: ", traversed_edge)
                break
            depth += 1
        if traversed_edge:
            if traversed_edge not in policy_bank.policies[policy_bank.get_id(ltl)].get_edge_labels():
                print("ERROR: traversed edge not an outgoing edge: ", traversed_edge)
            edge2hits[traversed_edge] += 1
    print(edge2hits)
    return edge2hits


def construct_initiation_set_classifiers(saver):
    """
    Map edge-centric option policy to its initiation set classifier.
    Classifier (policy2edge2loc2prob) contain only outgoing edges that state-centric policies achieved during training,
    possibly not all outgoing edges.
    """
    with open(os.path.join(saver.classifier_dpath, "rollout_results_parallel.pkl"), "rb") as rf:
        policy2loc2edge2hits = dill.load(rf)

    n_rollouts = policy2loc2edge2hits["n_rollouts"]
    policy2edge2loc2prob = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float)))
    policy2edge2loc2prob_json = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float)))
    for key, val in policy2loc2edge2hits.items():
        if key in ["n_rollouts", "ltls"]:
            continue
        ltl, loc2edge2hits = key, val
        print("ltl: ", ltl)
        for loc, edge2hits in loc2edge2hits.items():
            print("loc: ", loc)
            for edge, hits in edge2hits.items():
                prob = hits / n_rollouts
                policy2edge2loc2prob[ltl][edge][loc] = prob
                policy2edge2loc2prob_json[str(ltl)][str(edge)][str(loc)] = prob
    with open(os.path.join(saver.classifier_dpath, "classifier.json"), "w") as wf:
        json.dump(policy2edge2loc2prob_json, wf)  # save to json for easier inspection of dictionary

    return policy2edge2loc2prob


def zero_shot_transfer(tester, policy_bank, policy2edge2loc2prob):
    transfer_tasks = tester.get_transfer_tasks()
    edge2ltls, training_edges = get_training_edges(policy2edge2loc2prob)

    task2sol = defaultdict(list)

    # for transfer_task in transfer_tasks:
    #     print("transfer task:", transfer_task)
    #     task = Game(tester.get_task_params(transfer_task))  # same map as the training tasks
    #
    #     # wrapper: DFA -> NetworkX graph
    #     dfa_graph = dfa2graph(task.dfa)
    #
    #     # for edge, edge_data in dfa_graph.edges.items():
    #     #     print(edge, edge_data)
    #
    #     # pos = nx.circular_layout(dfa_graph)
    #     # nx.draw_networkx(dfa_graph, pos, with_labels=True)
    #     # nx.draw_networkx_edges(dfa_graph, pos)
    #     # ax = plt.gca()
    #     # ax.margins(0.20)
    #     # plt.axis("off")
    #     # plt.show()
    #
    #     # Graph search to find all paths from initial state to goal state
    #     all_simple_paths = nx.all_simple_paths(dfa_graph, source=task.dfa.state, target=task.dfa.terminal)
    #     all_simple_paths = [list(path) for path in map(nx.utils.pairwise, all_simple_paths)]
    #     print("start: ", task.dfa.state, "goal: ", task.dfa.terminal)
    #     print("all simple paths: ", len(all_simple_paths), all_simple_paths)
    #
    #     # Find all paths consists of only seen edges
    #     feasible_paths = []
    #     for simple_path in all_simple_paths:
    #         is_feasible_path = True
    #         for edge in simple_path:
    #             if not match_edges(dfa_graph.edges[edge[0], edge[1]]["edge_label"], training_edges):
    #                 is_feasible_path = False
    #                 break
    #         if is_feasible_path:
    #             feasible_paths.append(simple_path)

        # while not task.ltl_game_over and not task.env_game_over:
        #     cur_node = task.dfa.state
            # Find all paths from current node

            # Find 1st edge to target based on success probability from current MDP state

            # Execute option

            # task2sol[transfer_task].append(option)

    return task2sol


def get_training_edges(policy2edge2loc2prob):
    """
    Get all outgoing edges that state-centric policies have achieved during training.
    Map edge to corresponding LTLs, possibly one to many.
    """
    edge2ltls = defaultdict(list)
    training_edges = []
    for ltl, edge2ltl2loc2prob in policy2edge2loc2prob.items():
        for edge, _ in edge2ltl2loc2prob.items():
            edge2ltls[edge].append(ltl)
            if edge not in training_edges:
                training_edges.append(edge)

    # print("n_edges: ", len(edge2ltls))
    # for edge, ltls in edge2ltls.items():
    #     print("edge: ", edge)
    #     for ltl in ltls:
    #         print("ltl: ", ltl)
    #     print()

    # ltl2edges = {ltl: policy_bank.policies[policy_bank.get_id(ltl)].get_edge_labels() for ltl in policy_bank.get_LTL_policies()}
    # edge2ltls = defaultdict(list)
    # training_edges = []
    # for ltl in policy_bank.get_LTL_policies():
    #     out_edges = policy_bank.policies[policy_bank.get_id(ltl)].get_edge_labels()
    #     for edge in out_edges:
    #         edge2ltls[edge].append(ltl)  # all policies that can satisfy this edge
    #         if edge not in training_edges:
    #             training_edges.append(edge)
            # else:
            #     print("duplicate edge: ", ltl, edge)
    # pprint(edge2ltls)
    # print("training edges: ", len(training_edges), training_edges)
    return edge2ltls, training_edges


def dfa2graph(dfa):
    """
    convert DFA to a NetworkX graph
    """
    nodelist = defaultdict(dict)
    for u in dfa.nodelist:
        for v, label in dfa.nodelist[u].items():
            nodelist[u][v] = {"edge_label": label}
    dfa_graph = nx.DiGraph(nodelist)
    return dfa_graph


def match_edges(test_edge, training_edges):
    return True


def execute_option(task, policy_bank, ltl_goal):
    num_features = task.get_num_features()

    total_reward = 0
    # while not termination condition and policy is defined in current MDP state
    s1 = task.get_features()
    a = Actions(policy_bank.get_best_action(ltl_goal, s1.reshape((1, num_features))))
    reward = task.execute_action(a)
    total_reward += reward

    return total_reward