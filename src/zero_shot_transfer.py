import os
import re
import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from multiprocessing import Pool
import numpy as np
import sympy
import networkx as nx
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from lpopl import _initialize_policy_bank, _test_LPOPL
from policy_bank import *
from dfa import *
from game import *
from test_utils import Loader, save_pkl, load_pkl, save_json
from run_single_worker import single_worker_rollouts

CHUNK_SIZE = 441


def run_experiments(tester, curriculum, saver, run_id, relabel_method, num_times):
    loader = Loader(saver)

    time_init = time.time()
    learning_params = tester.learning_params

    random.seed(run_id)
    sess = tf.Session()

    # Reseting default values
    curriculum.restart()

    # Initializing policies per each subtask
    policy_bank = _initialize_policy_bank(sess, learning_params, curriculum, tester, load_tf=False)
    # loader.load_policy_bank(run_id, sess)

    # task_aux = Game(tester.get_task_params(tester.get_LTL_tasks()[0]))
    # num_features = task_aux.get_num_features()
    # tester.run_test(-1, sess, _test_LPOPL, policy_bank, num_features)  # -1 to signal test after restore models
    # print(tester.results)

    # Save LTL formula to ID mapping
    ltl2id_pkl_fpath = os.path.join(saver.classifier_dpath, "ltl2id_%d.pkl" % tester.train_size)
    if not os.path.exists(ltl2id_pkl_fpath):
        ltl2id_pkl = {}
        ltl2id_json = {}
        for ltl in policy_bank.get_LTL_policies():
            ltl_id = policy_bank.get_id(ltl)
            ltl2id_pkl[ltl] = ltl_id
            ltl2id_json[str(ltl)] = ltl_id
        save_pkl(ltl2id_pkl_fpath, ltl2id_pkl)
        save_json(os.path.join(saver.classifier_dpath, "ltl2id_%d.json" % tester.train_size), ltl2id_json)

    # Relabel state-centric options to transition-centric options if not already done it
    if not os.path.exists(os.path.join(saver.classifier_dpath, "aggregated_rollout_results.pkl")):
        if relabel_method == 'cluster':  # use mpi
            relabel_cluster(tester, saver, curriculum, run_id, policy_bank)
        if relabel_method == 'parallel':  # use Python multiprocessing
            relabel_parallel(tester, saver, curriculum, run_id, policy_bank)

    start_time = time.time()
    policy2edge2loc2prob = construct_initiation_set_classifiers(saver.classifier_dpath, policy_bank, tester.train_size)
    print("took %0.2f mins to construct inititation set classifier" % ((time.time() - start_time)/60))
    zero_shot_transfer(tester, policy_bank, loader, run_id, sess, policy2edge2loc2prob, num_times, curriculum.num_steps)

    tf.reset_default_graph()
    sess.close()

    # Log transfer results
    # tester.log_results("Transfer took: %0.2f mins\n" % ((time.time() - time_init)/60))
    print("Transfer took: %0.2f mins\n\n" % ((time.time() - time_init) / 60))
    saver.save_transfer_results()


def relabel_cluster(tester, saver, curriculum, run_id, policy_bank, n_rollouts=100):
    """
    A worker runs n_rollouts from a specific location for all LTL formulas in policy_bank
    """
    print('RELABELING STATE CENTRIC OPTIONS')
    task_aux = Game(tester.get_task_params(tester.get_LTL_tasks()[0]))
    id2ltls = {}
    for ltl in policy_bank.get_LTL_policies():
        ltl_id = policy_bank.get_id(ltl)
        id2ltls[ltl_id] = (ltl, policy_bank.policies[ltl_id].f_task)
    state2id = saver.save_transfer_data(task_aux, id2ltls)
    all_locs = [(x, y) for x in range(task_aux.map_width) for y in range(task_aux.map_height)]
    loc_chunks = [all_locs[chunk_id: chunk_id + CHUNK_SIZE] for chunk_id in range(0, len(all_locs), CHUNK_SIZE)]
    completed_ltls = []
    if os.path.exists(os.path.join(saver.classifier_dpath, "completed_ltls.pkl")):
        old_list = load_pkl(os.path.join(saver.classifier_dpath, "completed_ltls.pkl"))
        completed_ltls.extend(old_list)

    for idx, ltl in enumerate(policy_bank.get_LTL_policies()):
        ltl_id = policy_bank.get_id(ltl)
        if ltl_id in completed_ltls:
            continue  # Check if this formula was already compiled. If so continue to next formula

        start_time_ltl = time.time()
        print("index %s. Starting LTL: %s, %s" % (idx, ltl_id, ltl))

        for chunk_id, locs in enumerate(loc_chunks):
            args = []
            for x, y in locs:
                if task_aux.is_valid_agent_loc(x, y):
                    # create command to run a single worker
                    arg = (saver.alg_name, tester.train_type, tester.map_id, run_id, ltl_id, state2id[(x, y)], n_rollouts, curriculum.num_steps)
                    args.append(arg)
            args_len = len(args)
            # args2 = deepcopy(args)

            if args:
                start_time_chunk = time.time()
                with MPIPoolExecutor(max_workers=CHUNK_SIZE) as pool:  # parallelize over all locs in a chunk
                    retvals = pool.starmap(run_single_worker_cluster, args)
                for retval, arg in zip(retvals, args):
                    if retval:  # os.system exit code 0 means correct execution
                        print("Command failed: ", retval, arg)
                        retval = run_single_worker_cluster(*arg)
                print("chunk %s took: %0.2f mins, with %d states" % (chunk_id, (time.time() - start_time_chunk) / 60, args_len))
        print("Completed LTL %s took: %0.2f mins" % (ltl_id, (time.time() - start_time_ltl) / 60))
        completed_ltls.append(ltl_id)
        save_pkl(os.path.join(saver.classifier_dpath, "completed_ltls.pkl"), completed_ltls)
        save_json(os.path.join(saver.classifier_dpath, "completed_ltls.json"), completed_ltls)
    aggregate_rollout_results(task_aux, saver, policy_bank, n_rollouts)


def run_single_worker_cluster(algo, train_type, map_id, run_id, ltl_id, state_id, n_rollouts, max_depth):
    classifier_dpath = os.path.join("../tmp/", "%s/map_%d" % (train_type, map_id), "classifier")
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    # print(f"Running state {state_id} through process {rank} on {name}")
    single_worker_rollouts(algo, classifier_dpath, run_id, ltl_id, state_id, n_rollouts, max_depth)
    return 0


def relabel_parallel(tester, saver, curriculum, run_id, policy_bank, n_rollouts=100):
    """
    A worker runs n_rollouts from a specific location for each LTL formula in policy_bank
    """
    task_aux = Game(tester.get_task_params(tester.get_LTL_tasks()[0]))
    id2ltls = {}
    for ltl in policy_bank.get_LTL_policies():
        ltl_id = policy_bank.get_id(ltl)
        id2ltls[ltl_id] = (ltl, policy_bank.policies[ltl_id].f_task)
    state2id = saver.save_transfer_data(task_aux, id2ltls)
    all_locs = [(x, y) for x in range(task_aux.map_width) for y in range(task_aux.map_height)]
    loc_chunks = [all_locs[chunk_id: chunk_id + CHUNK_SIZE] for chunk_id in range(0, len(all_locs), CHUNK_SIZE)]
    completed_ltls = []
    if os.path.exists(os.path.join(saver.classifier_dpath, "completed_ltls.pkl")):
        old_list = load_pkl(os.path.join(saver.classifier_dpath, "completed_ltls.pkl"))['completed_ltl']
        completed_ltls.extend(old_list)

    for idx, ltl in enumerate(policy_bank.get_LTL_policies()):
        ltl_id = policy_bank.get_id(ltl)
        if ltl_id in completed_ltls:
            continue  # Check if this formula was already compiled. If so continue to next formula

        # if ltl_id not in [17]:
        #     continue
        start_time_ltl = time.time()
        print("index %s. Starting LTL: %s, %s" % (idx, ltl_id, ltl))

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
                    # create command to run a single worker
                    args = ("--algo=%s --train_type=%d --map_id=%d --run_id=%d --ltl_id=%d --state_id=%d --n_rollouts=%d --max_depth=%d" % (
                            saver.alg_name, tester.train_type, tester.map_id, run_id, ltl_id, state2id[(x, y)], n_rollouts, curriculum.num_steps))
                    worker_commands.append("python3 run_single_worker.py %s" % args)
            if worker_commands:
                start_time_chunk = time.time()
                with Pool(processes=len(worker_commands)) as pool:
                    retvals = pool.map(os.system, worker_commands)
                for retval, worker_command in zip(retvals, worker_commands):
                    if retval:  # os.system exit code 0 means correct execution
                        print("Command failed: ", retval, worker_command)
                        retval = os.system(worker_command)
                print("chunk %s took: %0.2f mins, with %d states" % (chunk_id, (time.time() - start_time_chunk) / 60, len(retvals)))
        print("Completed LTL %s took: %0.2f mins" % (ltl_id, (time.time() - start_time_ltl) / 60))
        completed_ltls.append(ltl_id)
        save_pkl(os.path.join(saver.classifier_dpath, "completed_ltls.pkl"), completed_ltls)
        save_json(os.path.join(saver.classifier_dpath, "completed_ltls.json"), completed_ltls)
    aggregate_rollout_results(task_aux, saver, policy_bank, n_rollouts)


def aggregate_rollout_results(task_aux, saver, policy_bank, n_rollouts):
    """
    Aggregate results saved locally by parallel workers for learning classifiers
    """
    policy2loc2edge2hits_pkl = {"n_rollouts": n_rollouts}
    policy2loc2edge2hits_json = {"n_rollouts": n_rollouts}
    for ltl in policy_bank.get_LTL_policies():
        ltl_id = policy_bank.get_id(ltl)
        # if ltl_id not in [17]:
        #     continue
        policy2loc2edge2hits_pkl[ltl] = {}
        policy2loc2edge2hits_json[str(ltl)] = {}
        for x in range(task_aux.map_width):
            for y in range(task_aux.map_height):
                # if (x, y) not in [(9, 2), (3, 11)]:
                #     continue
                if task_aux.is_valid_agent_loc(x, y):
                    worker_fpath = os.path.join(saver.classifier_dpath, "ltl%d_state%d-%d_" % (ltl_id, x, y))
                    rollout_results = load_pkl(worker_fpath+"rollout_results_parallel.pkl")
                    # try:  # for local testing parallel rollout with a few random locs
                    #     rollout_results = load_pkl(worker_fpath + "rollout_results_parallel.pkl")
                    # except IOError:
                    #     continue
                    policy2loc2edge2hits_pkl[ltl][(x, y)] = rollout_results["edge2hits"]
                    policy2loc2edge2hits_json[str(ltl)][str((x, y))] = rollout_results["edge2hits"]
    saver.save_rollout_results("aggregated_rollout_results", policy2loc2edge2hits_pkl, policy2loc2edge2hits_json)
    # Remove single worker rollout results to save space after aggregation
    for fname in os.listdir(saver.classifier_dpath):
        if re.match("ltl[0-9]+_state*", fname):
            os.remove(os.path.join(saver.classifier_dpath, fname))


def construct_initiation_set_classifiers(classifier_dpath, policy_bank, train_size):
    """
    Map edge-centric option policy to its initiation set classifier.
    Classifier (policy2edge2loc2prob) contain only outgoing edges that state-centric policies achieved during training,
    possibly not all outgoing edges.
    """
    classifier_pkl_fpath = os.path.join(classifier_dpath, "classifier_%d.pkl" % train_size)
    if os.path.exists(classifier_pkl_fpath):
        policy2edge2loc2prob = load_pkl(classifier_pkl_fpath)
    else:
        policy2loc2edge2hits = load_pkl(os.path.join(classifier_dpath, "aggregated_rollout_results.pkl"))
        n_rollouts = policy2loc2edge2hits["n_rollouts"]

        policy2edge2loc2prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        policy2edge2loc2prob_json = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for key, val in policy2loc2edge2hits.items():
            if key not in ["n_rollouts", "ltls"] and key in policy_bank.get_LTL_policies():  # only extract classifiers of the 'train_size' training policies
                ltl, loc2edge2hits = key, val
                for loc, edge2hits in loc2edge2hits.items():
                    for edge, hits in edge2hits.items():
                        prob = hits / n_rollouts
                        policy2edge2loc2prob[ltl][edge][loc] = prob
                        policy2edge2loc2prob_json[str(ltl)][str(edge)][str(loc)] = prob
        save_json(os.path.join(classifier_dpath, "classifier_%d.json" % train_size), policy2edge2loc2prob_json)
        save_pkl(classifier_pkl_fpath, policy2edge2loc2prob)

        edges2ltls_fpath = os.path.join(classifier_dpath, "edges2ltls_%d.txt" % train_size)
        if not os.path.exists(edges2ltls_fpath):
            _, edges2ltls = get_training_edges(policy_bank, policy2edge2loc2prob)
            with open(edges2ltls_fpath, "w") as wf:
                for count, (edge, ltls) in enumerate(edges2ltls.items()):
                    wf.write("(self_edge, outgoing_edge) %d: %s\n" % (count, str(edge)))
                    for ltl in ltls:
                        wf.write("ltl: %s\n" % str(ltl))
                    wf.write("\n")
    return policy2edge2loc2prob


def zero_shot_transfer(tester, policy_bank, loader, run_id, sess, policy2edge2loc2prob, num_times, num_steps):
    transfer_tasks = tester.get_transfer_tasks()
    train_edges, edge2ltls = get_training_edges(policy_bank, policy2edge2loc2prob)

    for task_idx, transfer_task in enumerate(transfer_tasks):
        tester.log_results("== Transfer Task %d: %s" % (task_idx, str(transfer_task)))
        print("== Transfer Task %d: %s\n" % (task_idx, str(transfer_task)))

        task_aux = Game(tester.get_task_params(transfer_task))  # same grid map as the training tasks
        # Wrapper: DFA -> NetworkX graph
        dfa_graph = dfa2graph(task_aux.dfa)
        for line in nx.generate_edgelist(dfa_graph):
            print(line)

        print("\ntraining edges: ", train_edges)
        # Remove edges in DFA that do not have a matching train edge
        start_time = time.time()
        test2trains = remove_infeasible_edges(dfa_graph, train_edges, task_aux.dfa.state, task_aux.dfa.terminal[0])

        if not test2trains:
            tester.log_results("DFA graph disconnected after removing infeasible edges\n")
            print("DFA graph disconnected after removing infeasible edges\n")
            continue
        tester.log_results("took %0.2f mins to remove infeasible edges" % ((time.time() - start_time) / 60))
        print("took %0.2f mins to remove infeasible edges" % ((time.time() - start_time) / 60))
        print("\nNew DFA graph")
        for line in nx.generate_edgelist(dfa_graph):
            print(line)

        # Graph search to find all simple/feasible paths from initial state to goal state
        feasible_paths_node = list(nx.all_simple_paths(dfa_graph, source=task_aux.dfa.state, target=task_aux.dfa.terminal))
        feasible_paths_edge = [list(path) for path in map(nx.utils.pairwise, feasible_paths_node)]
        tester.log_results("\ndfa start: %d; goal: %s" % (task_aux.dfa.state, str(task_aux.dfa.terminal)))
        print("\ndfa start: %d; goal: %s" % (task_aux.dfa.state, str(task_aux.dfa.terminal)))
        print("feasible paths: %s\n" % str(feasible_paths_node))

        if not feasible_paths_node:
            tester.log_results("No feasible DFA paths found to achieve this transfer task\n")
            print("No feasible DFA paths found to achieve this transfer task\n")
            continue

        for num_time in range(num_times):
            tester.log_results("** Run %d. Transfer Task %d: %s" % (num_time, task_idx, str(transfer_task)))
            print("** Run %d. Transfer Task %d: %s\n" % (num_time, task_idx, str(transfer_task)))

            task = Game(tester.get_task_params(transfer_task))  # same grid map as the training tasks
            total_reward = 0
            while not task.ltl_game_over and not task.env_game_over:
                cur_node = task.dfa.state
                tester.log_results("\ncurrent node: %d" % cur_node)
                print("\ncurrent node: %d" % cur_node)
                # optimization: check what outgoing edges are on a feasible path
                # out_edges = [(cur_node, next_node) for next_node in dfa_graph.successors(cur_node)]

                # Find all feasible paths the current node is on then candidate option edges to target
                candidate_edges = set()
                for feasible_path_node, feasible_path_edge in zip(feasible_paths_node, feasible_paths_edge):
                    if cur_node in feasible_path_node:
                        pos = feasible_path_node.index(cur_node)  # current position on this path
                        test_edge = feasible_path_edge[pos]
                        self_edge = dfa_graph.edges[test_edge[0], test_edge[0]]["edge_label"]  # self_edge label
                        out_edge = dfa_graph.edges[test_edge]["edge_label"]  # get boolean formula for outgoing edge
                        test_edge_pair = (self_edge, out_edge)
                        candidate_edges.update(test2trains[test_edge_pair])
                print("candidate edges: %s\n" % str(candidate_edges))

                # Find best edge to target based on rollout success probability from current location
                option2prob = {}
                cur_loc = (task.agent.i, task.agent.j)
                next_loc = cur_loc
                for self_edge, out_edge in candidate_edges:
                    for ltl in edge2ltls[(self_edge, out_edge)]:
                        option2prob[(ltl, self_edge, out_edge)] = policy2edge2loc2prob[ltl][out_edge][cur_loc]
                if not option2prob:
                    tester.log_results("No options found at DFA state %d, location %s\n" % (cur_node, str(cur_loc)))
                    print("No options found at DFA state %d, location %s\n" % (cur_node, str(cur_loc)))
                    break
                while option2prob and cur_loc == next_loc:
                    best_policy, best_self_edge, best_out_edge = sorted(option2prob.items(), key=lambda kv: kv[1])[-1][0]
                    # Overwrite empty policy by policy with tf model then load its weights when need to execute it
                    policy = policy_bank.policies[policy_bank.policy2id[best_policy]]
                    if not policy.load_tf:
                        policy_bank.replace_policy(policy.ltl, policy.f_task, policy.dfa)
                        loader.load_policy_bank(run_id, sess)
                    # Execute option
                    tester.log_results("executing option edge: (%s, %s)" % (str(best_self_edge), str(best_out_edge)))
                    print("executing option edge: (%s, %s)" % (str(best_self_edge), str(best_out_edge)))
                    tester.log_results("from policy %d: %s" % (policy_bank.get_id(best_policy), str(best_policy)))
                    print("from policy %d: %s" % (policy_bank.get_id(best_policy), str(best_policy)))
                    next_loc, option_reward = execute_option(tester, task, policy_bank, best_policy, best_out_edge, policy2edge2loc2prob[best_policy], num_steps)
                    if cur_loc != next_loc:
                        tester.task2run2sol[str(transfer_task)][num_time].append((str(best_policy), best_self_edge, best_out_edge))
                        total_reward += option_reward
                        tester.log_results("option changed loc: %s; option_reward: %d\n" % (str(cur_loc != next_loc), option_reward))
                        print("option changed loc: %s; option_reward: %d\n" % (str(cur_loc != next_loc), option_reward))
                    else:  # if best option did not change agent location, try second best option
                        print(option2prob)
                        print(cur_loc, next_loc)
                        del option2prob[(best_policy, best_self_edge, best_out_edge)]
                if cur_loc == next_loc:
                    tester.log_results("No options progress LTL from DFA state %d, location %s\n" % (cur_node, str(cur_loc)))
                    print("No options progress LTL from DFA state %d, location %s\n" % (cur_node, str(cur_loc)))
                    break
            tester.log_results("current node: %d\n\n" % task.dfa.state)
            print("current node: %d\n\n" % task.dfa.state)
            if task.ltl_game_over:
                tester.task2success[str(transfer_task)] += 1
    tester.task2success = {task: success/num_times for task, success in tester.task2success.items()}


def get_training_edges(policy_bank, policy2edge2loc2prob):
    """
    Pair every outgoing edge that each state-centric policy have achieved during training,
    with the self-edge of the DFA progress state corresponding to the state-centric policy.
    Map each edge pair to corresponding LTLs, possibly one to many.
    """
    edges2ltls = defaultdict(list)
    for ltl, edge2loc2prob in policy2edge2loc2prob.items():
        dfa = policy_bank.policies[policy_bank.get_id(ltl)].dfa
        node = dfa.ltl2state[ltl]
        self_edge = dfa.nodelist[node][node]
        for out_edge in edge2loc2prob.keys():
            edges2ltls[(self_edge, out_edge)].append(ltl)
    return edges2ltls.keys(), edges2ltls


def dfa2graph(dfa):
    """
    Convert DFA to NetworkX graph
    """
    nodelist = defaultdict(dict)
    for u, v2label in dfa.nodelist.items():
        for v, label in v2label.items():
            nodelist[u][v] = {"edge_label": label}
    return nx.DiGraph(nodelist)


def remove_infeasible_edges(dfa, train_edges, start_state, goal_state):
    """
    Remove infeasible edges from DFA graph, and
    construct a mapping from test_edge_pair to a set of matching train_edge_pairs
    """
    test2trains = defaultdict(set)
    edges = list(dfa.edges.keys())  # make a copy of edges because nx graph dict changes
    for edge in edges:
        if edge[0] != edge[1]:  # ignore self-edge
            self_edge_label = dfa.edges[edge[0], edge[0]]["edge_label"]
            out_edge_label = dfa.edges[edge]["edge_label"]
            test_edge_pair = (self_edge_label, out_edge_label)
            is_matched = False
            for train_edge in train_edges:
                if match_edges(test_edge_pair, [train_edge]):
                    test2trains[test_edge_pair].add(train_edge)
                    is_matched = True
            if not is_matched:
                dfa.remove_edge(edge[0], edge[1])
                # optimization: return as soon as start and goal are not connected
                feasible_paths_node = list(nx.all_simple_paths(dfa, source=start_state, target=goal_state))
                if not feasible_paths_node:
                    print("short circuit remove_infeasible_edges")
                    return None
    return test2trains


def match_edges(test_edge_pair, train_edges):
    """
    Determine if test_edge can be matched with any training_edge
    match := exact match (aka. eq) or test_edge is less constrained than a training_edge (aka. subset)
    Note: more efficient to convert 'training_edges' before calling this function
    """
    test_self_edge, test_out_edge = test_edge_pair
    test_self_edge = sympy.simplify_logic(test_self_edge.replace('!', '~'), form='dnf')
    test_out_edge = sympy.simplify_logic(test_out_edge.replace('!', '~'), form='dnf')
    train_self_edges = [sympy.simplify_logic(pair[0].replace('!', '~'), form='dnf') for pair in train_edges]
    train_out_edges = [sympy.simplify_logic(pair[1].replace('!', '~'), form='dnf') for pair in train_edges]

    is_subset_eq_self = np.any([bool(_is_subset_eq(test_self_edge, train_self_edge)) for train_self_edge in train_self_edges])
    is_subset_eq_out = np.any([bool(_is_subset_eq(test_out_edge, train_out_edge)) for train_out_edge in train_out_edges])
    return is_subset_eq_self and is_subset_eq_out


def _is_subset_eq(test_edge, train_edge):
    """
    subset_eq match :=
    every conjunctive term of 'test_edge' can be satisfied by the same 'training_edge'

    Assume edges are in sympy and DNF
    DNF: negation can only precede a propositional variable
    e.g. ~a | b is DNF; ~(a & b) is not DNF

    https://github.com/sympy/sympy/issues/23167
    """
    if test_edge.func == sympy.Or:
        if train_edge.func == sympy.Or:  # len(train_edge.args) <= len(test_edge.args)
            return sympy.And(*[sympy.Or(*[_is_subset_eq(test_term, train_term) for test_term in test_edge.args])
                               for train_term in train_edge.args]) and \
                   sympy.And(*[sympy.Or(*[_is_subset_eq(test_term, train_term) for train_term in train_edge.args])
                               for test_term in test_edge.args])
        return sympy.Or(*[_is_subset_eq(term, train_edge) for term in test_edge.args])
    elif test_edge.func == sympy.And:
        return train_edge.func == sympy.And and sympy.And(*[term in train_edge.args for term in test_edge.args])
    else:  # Atom, e.g. a, b, c or Not
        if train_edge.func == sympy.And:
            return test_edge in train_edge.args
        else:
            return test_edge == train_edge


def execute_option(tester, task, policy_bank, ltl_policy, option_edge, edge2loc2prob, num_steps):
    """
    'option_edge' is 1 outgoing edge associated with edge-centric option
    'option_edge' maye be different from target DFA edge when 'option_edge' is more constraint than target DFA edge
    """
    num_features = task.get_num_features()
    option_reward, step = 0, 0
    cur_node, cur_loc = task.dfa.state, (task.agent.i, task.agent.j)
    tester.log_results("cur_loc: %s" % str(cur_loc))
    print("cur_loc: %s" % str(cur_loc))
    # while not exceed max steps AND no DFA transition occurs AND option policy is still defined in current MDP state
    while step < num_steps and cur_node == task.dfa.state and cur_loc in edge2loc2prob[option_edge]:
        cur_node = task.dfa.state
        s1 = task.get_features()
        a = Actions(policy_bank.get_best_action(ltl_policy, s1.reshape((1, num_features))))
        if task._get_next_position(a) not in edge2loc2prob[option_edge]:  # check if possible next loc in initiation set
            break
        option_reward += task.execute_action(a)
        tester.log_results("step %d: dfa_state: %d; %s; %s; %d" % (step, cur_node, str(cur_loc), str(a), option_reward))
        print("step %d: dfa_state: %d; %s; %s; %d" % (step, cur_node, str(cur_loc), str(a), option_reward))
        cur_loc = (task.agent.i, task.agent.j)
        step += 1
    return cur_loc, option_reward
