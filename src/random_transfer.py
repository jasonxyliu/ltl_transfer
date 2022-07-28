import os
import re
import time
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
try:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
except:
    print('MPI installation not found. Please do not use cluster computing options')
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

RELABEL_CHUNK_SIZE = 441
TRANSFER_CHUNK_SIZE = 100


def run_experiments(tester, curriculum, saver, run_id, relabel_method, num_times):
    loader = Loader(saver)

    time_init = time.time()
    learning_params = tester.learning_params

    random.seed(run_id)
    random_transfer_cluster(tester, loader, saver, run_id, num_times, curriculum.num_steps, learning_params, curriculum)

    tf.reset_default_graph()
    sess.close()

    # Log transfer results
    # tester.log_results("Transfer took: %0.2f mins\n" % ((time.time() - time_init)/60))
    print("Transfer took: %0.2f mins\n\n" % ((time.time() - time_init) / 60))
    saver.save_transfer_results()





def random_transfer_cluster(tester, loader, saver, run_id, num_times, num_steps, learning_params, curriculum):
    # Precompute common computations
    transfer_tasks = tester.get_transfer_tasks()
    ltl_ids = list(range(len(transfer_tasks)))

    task_chunks = [transfer_tasks[chunk_id: chunk_id + TRANSFER_CHUNK_SIZE] for chunk_id in range(0, len(transfer_tasks), TRANSFER_CHUNK_SIZE)]
    ltl_id_chunks = [ltl_ids[chunk_id: chunk_id + TRANSFER_CHUNK_SIZE] for chunk_id in range(0, len(ltl_ids), TRANSFER_CHUNK_SIZE)]

    # Define task parameters and arguments
    retvals = []
    for (chunk_id, (task_chunk, id_chunk)) in enumerate(zip(task_chunks, ltl_id_chunks)):
        args = []
        for (transfer_task, ltl_id) in zip(task_chunk, id_chunk):
            args.append((transfer_task, ltl_id, num_times, num_steps, run_id, learning_params, curriculum, tester, loader, saver))
        # Send tasks to parallel workers
        print(f'Starting transfer chunk {chunk_id} of {len(task_chunks)}')
        start_time = time.time()
        with MPIPoolExecutor(max_workers=TRANSFER_CHUNK_SIZE) as pool:  # parallelize over transfer tasks
            retvals_chunk = pool.starmap(random_transfer_single_task, args)
        retvals.extend(retvals_chunk)
        print(f'Completed transfer chunk {chunk_id} of {len(task_chunks)} in {(time.time() - start_time)/60} minutes')
        print(retvals)

        # Accumulate results
        for (transfer_task, retval) in zip(transfer_tasks, retvals):
            tester.task2success[str(transfer_task)] = retval[0]
            tester.task2run2sol[str(transfer_task)] = retval[1]
            tester.task2run2trajs[str(transfer_task)] = retval[2]
# unpicklable objects: train_edges (dict_keys), learning_params, curriculum, tester


def random_transfer_single_task(transfer_task, ltl_idx, num_times, num_steps, run_id, learning_params, curriculum, tester, loader, saver):
    print('Starting single worker transfer to task: %s' % str(transfer_task))
    logfilename = os.path.join(tester.transfer_results_dpath, f'test_ltl_{ltl_idx}.pkl')
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True)
    tf.reset_default_graph()
    
    n_horiz = 10 #Multiplier for single task policy horizon

    # Load the policy bank without loading the policies
    #policy_bank = _initialize_policy_bank(sess, learning_params, curriculum, tester, load_tf=False)
    #policy2edge2loc2prob = construct_initiation_set_classifiers(saver.classifier_dpath, policy_bank, tester.train_size)
    #train_edges, edge2ltls = get_training_edges(policy_bank, policy2edge2loc2prob)

    task_aux = Game(tester.get_task_params(transfer_task))
    #dfa_graph = dfa2graph(task_aux.dfa)

    success, run2sol, run2traj, run2exitcode = 0, defaultdict(list), {}, {}

    #start_time = time.time()
    start_time = time.time()
    for num_time in range(num_times):
        task = Game(tester.get_task_params(transfer_task))
        run_traj = []
        #node2option2prob = {}
        step = 0
        while not task.ltl_game_over and not task.env_game_over and step <= n_horiz*num_steps:
            cur_node = task.dfa.state
            #next_node = cur_node
            cur_loc = (task.agent.i, task.agent.j)
            
            actions = task._load_actions()
            a = np.random.choice(actions)
            r = task.execute_action(a)
            transition = ((cur_loc, cur_node), a.name, r, ((task.agent.i, task.agent.j), task.dfa.state))
            run_traj.append(transition)
            step = step+1
            
            if task.ltl_game_over:
                if task.dfa.state != -1:
                    success += 1
                    run2exitcode[num_time] = 0
                else:
                    run2exitcode[num_time] = 'specification_fail'
            
            if step > n_horiz*num_steps:
                run2exitcode[num_time] = 'timeout'

        run2traj[num_time] = run_traj
    success = success / num_times
    mean_run_time = (time.time() - start_time) / num_times
    runtime = mean_run_time

    # log single task result
    data = {'transfer_task': transfer_task, 'success': success, 'run2sol': run2sol, 'run2traj': run2traj, 'run2exitcode': run2exitcode, 'runtime': runtime}
    save_pkl(logfilename, data)

    print('Finished single worker transfer to task: %s' % str(transfer_task))
    return success, run2sol, run2traj, run2exitcode, runtime



if __name__ == '__main__':
    a=1
