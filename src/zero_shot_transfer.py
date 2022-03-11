import os
import time
import random
import json
import dill
from multiprocessing import Pool
from collections import defaultdict
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import networkx as nx
import sympy
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from lpopl import _initialize_policy_bank, _test_LPOPL
from policy_bank import *
from dfa import *
from game import *

CHUNK_SIZE = 32


def run_experiments(tester, curriculum, saver, loader, run_id):
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
    # relabel_parallel(tester, saver, curriculum, run_id, policy_bank)
    policy2edge2loc2prob = construct_initiation_set_classifiers(saver.classifier_dpath)
    task2sol = zero_shot_transfer(tester, policy_bank, policy2edge2loc2prob, 1, curriculum.num_steps)
    # saver.save_transfer_results()

    tf.reset_default_graph()
    sess.close()

    # Showing transfer results
    tester.show_transfer_results()
    print("Time: %0.2f mins" % ((time.time() - time_init)/60))


def relabel_parallel(tester, saver, curriculum, run_id, policy_bank, n_rollouts=100):
    """
    A worker runs n_rollouts from a specific location for all LTL formulas in policy_bank
    """
    # Save LTL formula to ID to mapping for inspection later
    ltl2id_pkl_fpath = os.path.join(saver.classifier_dpath, "ltl2id.pkl")
    if not os.path.exists(ltl2id_pkl_fpath):
        ltl2id_pkl = {}
        ltl2id_json = {}
        for ltl in policy_bank.get_LTL_policies():
            ltl_id = policy_bank.get_id(ltl)
            ltl2id_pkl[ltl] = ltl_id
            ltl2id_json[str(ltl)] = ltl_id
        with open(ltl2id_pkl_fpath, 'wb') as file:
            dill.dump(ltl2id_pkl, file)
        with open(os.path.join(saver.classifier_dpath, "ltl2id.json"), 'w') as file:
            json.dump(ltl2id_json, file)

    task_aux = Game(tester.get_task_params(tester.get_LTL_tasks()[0]))
    state2id = saver.save_training_data(task_aux)
    all_locs = [(x, y) for x in range(task_aux.map_width) for y in range(task_aux.map_height)]
    loc_chunks = [all_locs[chunk_id: chunk_id+CHUNK_SIZE] for chunk_id in range(0, len(all_locs), CHUNK_SIZE)]
    completed_ltls = []
    if os.path.exists(os.path.join(saver.classifier_dpath, "completed_ltls.pkl")):
        with open(os.path.join(saver.classifier_dpath, "completed_ltls.pkl"), 'rb') as file:
            old_list = dill.load(file)['completed_ltl']
        completed_ltls.extend(old_list)

    for ltl_idx, ltl in enumerate(policy_bank.get_LTL_policies()):
        ltl_id = policy_bank.get_id(ltl)

        if ltl_id in completed_ltls:
            continue  # Check if this formula was already compiled. If so continue to next formula

        # if ltl_id not in [17]:
        #     continue
        print("index ", ltl_idx, ". ltl (sub)task: ", ltl, ltl_id)
        start_time_ltl = time.time()
        print("Starting LTL: %s, %s, %s" % (ltl_id, ltl, ltl_idx))

        # x_tests = np.random.randint(1, 20, size=1)
        # y_tests = np.random.randint(1, 20, size=1)
        # test_locs = list(zip(x_tests, y_tests))
        # test_locs = [(9, 2), (3, 11)]
        # print("test_locs: ", test_locs)
        for chunk_id, locs in enumerate(loc_chunks):
            worker_commands = []
            for x, y in locs:
                # print(x, y)
                # if (x, y) not in test_locs:
                #     continue
                if task_aux.is_valid_agent_loc(x, y):
                    # create directory to store results from a single worker
                    # saver.create_worker_directory(ltl_id, state2id[(x, y)])
                    # create command to run a single worker
                    args = "--algo=%s --tasks_id=%d --map_id=%d --run_id=%d --ltl_id=%d --state_id=%d --n_rollouts=%d --max_depth=%d" % (
                        saver.alg_name, tester.tasks_id, tester.map_id, run_id, ltl_id, state2id[(x, y)], n_rollouts, curriculum.num_steps)
                    worker_commands.append("python3 run_single_worker.py %s" % args)
            # print(worker_commands)
            if worker_commands:
                start_time_chunk = time.time()
                with Pool(processes=len(worker_commands)) as pool:
                    retvals = pool.map(os.system, worker_commands)
                for retval, worker_command in zip(retvals, worker_commands):
                    if retval:  # os.system exit code: 0 means correct execution
                        print("Command failed: ", retval, worker_command)
                        retval = os.system(worker_command)
                print("chunk %s took: %0.2f, with %d states" % (chunk_id, (time.time() - start_time_chunk) / 60, len(retvals)))
        print("Completed LTL %s took: %0.2f" % (ltl_id, (time.time()-start_time_ltl)/60))
        completed_ltls.append(ltl_id)
        with open(os.path.join(saver.classifier_dpath, "completed_ltls.pkl"), 'wb') as file:
            dill.dump({'completed_ltl': completed_ltls}, file)
        with open(os.path.join(saver.classifier_dpath, "completed_ltls.json"), 'w') as file:
            json.dump(completed_ltls, file)

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
        # if ltl_id not in [17]:
        #     continue
        policy2loc2edge2hits_json[str(ltl)] = {}
        policy2loc2edge2hits_pkl[ltl] = {}
        for x in range(task_aux.map_width):
            for y in range(task_aux.map_height):
                # if (x, y) not in [(9, 2), (3, 11)]:
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


def construct_initiation_set_classifiers(classifier_dpath):
    """
    Map edge-centric option policy to its initiation set classifier.
    Classifier (policy2edge2loc2prob) contain only outgoing edges that state-centric policies achieved during training,
    possibly not all outgoing edges.
    """
    with open(os.path.join(classifier_dpath, "rollout_results_parallel.pkl"), "rb") as rf:
        policy2loc2edge2hits = dill.load(rf)

    n_rollouts = policy2loc2edge2hits["n_rollouts"]
    policy2edge2loc2prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    policy2edge2loc2prob_json = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for key, val in policy2loc2edge2hits.items():
        if key in ["n_rollouts", "ltls"]:
            continue
        ltl, loc2edge2hits = key, val
        # print("ltl: ", ltl)
        for loc, edge2hits in loc2edge2hits.items():
            # print("loc: ", loc)
            for edge, hits in edge2hits.items():
                prob = hits / n_rollouts
                policy2edge2loc2prob[ltl][edge][loc] = prob
                policy2edge2loc2prob_json[str(ltl)][str(edge)][str(loc)] = prob

    with open(os.path.join(classifier_dpath, "classifier.json"), "w") as wf:
        json.dump(policy2edge2loc2prob_json, wf)  # save to json for easier inspection of dictionary
    return policy2edge2loc2prob


def zero_shot_transfer(tester, policy_bank, policy2edge2loc2prob, num_times, num_steps):
    transfer_tasks = tester.get_transfer_tasks()
    training_edges, edge2ltls = get_training_edges(policy2edge2loc2prob)
    # training_edges = [sympy.simplify("c&~f"), sympy.simplify("f&~b"), sympy.simplify("b&h")]
    # print("num training edges: ", len(edge2ltls))
    # for edge, ltls in edge2ltls.items():
    #     print("edge: ", edge, type(edge))
    #     for ltl in ltls:
    #         print("ltl: ", ltl)
    #     print()

    task2sol = defaultdict(list)
    for transfer_task in transfer_tasks:
        # Running each transfer task 'num_times'
        for num_time in range(num_times):
            print("transfer task:", transfer_task)
            task = Game(tester.get_task_params(transfer_task))  # same grid map as the training tasks

            # Wrapper: DFA -> NetworkX graph
            dfa_graph = dfa2graph(task.dfa)
            # for edge, edge_data in dfa_graph.edges.items():
            #     print(edge, edge_data)
            # pos = nx.circular_layout(dfa_graph)
            # nx.draw_networkx(dfa_graph, pos, with_labels=True)
            # nx.draw_networkx_edges(dfa_graph, pos)
            # ax = plt.gca()
            # ax.margins(0.20)
            # plt.axis("off")
            # plt.show()

            # Graph search to find all simple paths from initial state to goal state
            simple_paths_node = list(nx.all_simple_paths(dfa_graph, source=task.dfa.state, target=task.dfa.terminal))
            simple_paths_edge = [list(path) for path in map(nx.utils.pairwise, simple_paths_node)]
            print("start: %d; goal: %s" % (task.dfa.state, str(task.dfa.terminal)))
            print("simple paths: ", len(simple_paths_node), simple_paths_node)

            # Find all paths consists of only edges matching training edges
            feasible_paths_node, feasible_paths_edge = feasible_paths(dfa_graph, simple_paths_node, simple_paths_edge, training_edges)
            print("feasible paths: ", feasible_paths_node, "\n")

            total_reward = 0
            while not task.ltl_game_over and not task.env_game_over:
                cur_node = task.dfa.state
                print("current node: ", cur_node)
                # Find all feasible paths the current node is on then candidate edges to target
                candidate_edges = []
                for feasible_path_node, feasible_path_edge in zip(feasible_paths_node, feasible_paths_edge):
                    print("feasible path: ", feasible_path_node)
                    if cur_node in feasible_path_node:
                        pos = feasible_path_node.index(cur_node)  # current position on the path
                        candidate_edges.append(feasible_path_edge[pos])
                        print("current position on a feasible path: ", pos)
                        print("candidate target edge: ", feasible_path_edge[pos])
                print("candidate edges: ", candidate_edges, "\n")
                # Find best edge to target based on rollout success probability from current location
                cur_loc = (task.agent.i, task.agent.j)
                option2prob = {}
                for edge in candidate_edges:
                    edge_label = dfa_graph.edges[edge]["edge_label"]  # get boolean formula for this edge
                    for ltl in edge2ltls[edge_label]:
                        option2prob[(ltl, edge_label)] = policy2edge2loc2prob[ltl][edge_label][cur_loc]
                if option2prob:
                    # for policy, prob in option2prob.items():
                    #     print(policy[0])
                    #     print(policy[1])
                    #     print(prob, "\n")
                    best_policy, best_edge_label = sorted(option2prob.items(), key=lambda kv: kv[1])[-1][0]
                    # Execute option
                    is_option_success, option_reward = execute_option(task, dfa_graph, policy_bank, best_policy, best_edge_label, policy2edge2loc2prob, num_steps)
                    print(is_option_success, option_reward, "\n")
                    if is_option_success:
                        task2sol[transfer_task].append((best_policy, best_edge_label))  # add option to solution
                        total_reward += option_reward
                else:
                    print("option2prob: ", option2prob)
                    print("No options found to achieve for task %s\n from DFA state %d, location %s" % (str(transfer_task), cur_node, str(cur_loc)))
                    break
    return task2sol


def get_training_edges(policy2edge2loc2prob):
    """
    Get all outgoing edges that state-centric policies have achieved during training.
    Map edge to corresponding LTLs, possibly one to many.
    """
    edge2ltls = defaultdict(list)
    for ltl, edge2loc2prob in policy2edge2loc2prob.items():
        for edge, _ in edge2loc2prob.items():
            edge2ltls[edge].append(ltl)
    training_edges = [sympy.simplify_logic(edge.replace('!', '~'), form='dnf') for edge in edge2ltls.keys()]
    return training_edges, edge2ltls


def dfa2graph(dfa):
    """
    Convert DFA to NetworkX graph
    """
    nodelist = defaultdict(dict)
    for u, v2label in dfa.nodelist.items():
        for v, label in v2label.items():
            nodelist[u][v] = {"edge_label": label}
    return nx.DiGraph(nodelist)


def feasible_paths(dfa_graph, all_simple_paths_node, all_simple_paths_edge, training_edges):
    """
    A feasible path consists of only DFA edges seen in training
    """
    feasible_paths_node = []
    feasible_paths_edge = []
    for simple_path_node, simple_path_edge in zip(all_simple_paths_node, all_simple_paths_edge):
        # print("path: %s\n" % str(simple_path_edge))
        is_feasible_path = True
        for edge in simple_path_edge:
            if not match_edges(dfa_graph.edges[edge]["edge_label"], training_edges):
                is_feasible_path = False
                break
        if is_feasible_path:
            feasible_paths_node.append(simple_path_node)
            feasible_paths_edge.append(simple_path_edge)
    return feasible_paths_node, feasible_paths_edge


def match_edges(test_edge, training_edges):
    """
    Determine if test_edge can be matched with any training_edge
    match := exact match or test_edge is less constrained than a training_edge, aka. subset

    Assume training_edges are expressed in sympy and simplified
    """
    test_edge_dnf = sympy.simplify_logic(test_edge.replace('!', '~'), form='dnf')
    is_exact_match = test_edge_dnf in training_edges
    is_subset = np.any([bool(_is_subset(test_edge_dnf, training_edge_dnf)) for training_edge_dnf in training_edges])
    return is_exact_match or is_subset


def _is_subset(test_edge, training_edge):
    """
    subset match :=
    every conjunctive term of test_edge can be satisfied by the same training_edge

    Assume edges are in DNF
    DNF: negation can only precede a propositional variable
    e.g. ~a | b is DNF; ~(a & b) is not DNF

    https://github.com/sympy/sympy/issues/23167
    """
    # print("test_edge: ", test_edge)
    # print("training_edge: ", training_edge)
    if test_edge.func == sympy.Or:
        if training_edge.func == sympy.Or:
            return sympy.And(*[sympy.Or(*[_is_subset(test_term, train_term) for test_term in test_edge.args])
                               for train_term in training_edge.args]) and \
                   sympy.And(*[sympy.Or(*[_is_subset(test_term, train_term) for train_term in training_edge.args])
                               for test_term in test_edge.args])
        return sympy.Or(*[_is_subset(term, training_edge) for term in test_edge.args])
    elif test_edge.func == sympy.And:
        return training_edge.func == sympy.And and sympy.And(*[term in training_edge.args for term in test_edge.args])
    else:  # Atom, e.g. a, b, c or Not
        if training_edge.func == sympy.And:
            return test_edge in training_edge.args
        else:
            return test_edge == training_edge


def execute_option(task, dfa_graph, policy_bank, ltl_policy, edge_label, policy2edge2loc2prob, num_steps):
    print("policy: ", ltl_policy)
    print("edge: ", edge_label)
    num_features = task.get_num_features()
    is_option_success, option_reward, step = False, 0, 0
    cur_node, cur_loc = task.dfa.state, (task.agent.i, task.agent.j)
    # while not exceed max steps and no DFA transition occurs and option policy is still defined in current MDP state
    while step < num_steps and cur_node == task.dfa.state and cur_loc in policy2edge2loc2prob[ltl_policy][edge_label]:
        cur_node = task.dfa.state
        s1 = task.get_features()
        a = Actions(policy_bank.get_best_action(ltl_policy, s1.reshape((1, num_features))))
        option_reward += task.execute_action(a)
        print("step %d: dfa_state: %d; %s; %s; %d" % (step, cur_node, str(cur_loc), str(a), option_reward))
        cur_loc = (task.agent.i, task.agent.j)
        step += 1
    if dfa_graph.edges[cur_node, task.dfa.state]["edge_label"] == edge_label:  # desired edge transition occurs
        is_option_success = True
    return is_option_success, option_reward
