import os
import dill
import argparse
from collections import defaultdict
import tensorflow as tf
from test_utils import Saver, Loader
from game import *
from policy_bank import *


def initialize_policy_bank(sess, task_aux, tester):
    num_actions  = len(task_aux.get_actions())
    num_features = task_aux.get_num_features()
    policy_bank = PolicyBank(sess, num_actions, num_features, tester.learning_params)
    for f_task in tester.get_LTL_tasks():
        dfa = DFA(f_task)
        for ltl in dfa.ltl2state:
            # this method already checks that the policy is not in the bank and it is not 'True' or 'False'
            policy_bank.add_LTL_policy(ltl, f_task, dfa)
    policy_bank.reconnect()  # -> creating the connections between the neural nets

    # print("\n", policy_bank.get_number_LTL_policies(), "sub-tasks were extracted!\n")
    return policy_bank


def single_worker_rollouts(alg_name, classifier_dpath, run_id, ltl_id, state_id, n_rollouts, max_depth):
    """
    Rollout a trained state-centric policy from init_state to see which outgoing edge it satisfies
    """
    # load tester
    with open(os.path.join(classifier_dpath, "tester.pkl"), "rb") as file:
        tester = dill.load(file)
    saver = Saver(alg_name, tester)
    loader = Loader(saver)
    # print("policy_dpath in worker: ", loader.saver.policy_dpath)

    # load init_state
    with open(os.path.join(classifier_dpath, "states.pkl"), "rb") as file:
        id2state = dill.load(file)
    init_state = id2state[state_id]
    # print("init_state: ", init_state, state_id)

    # create task_aux
    task_aux = Game(tester.get_task_params(tester.get_LTL_tasks()[0]))

    # ensure that tensorflow threads are restricted to a single core
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # load policy_bank
        # print("loading policy bank")
        policy_bank = initialize_policy_bank(sess, task_aux, tester)
        loader.load_policy_bank(run_id, sess)

        id2ltl = {pid: policy for policy, pid in policy_bank.policy2id.items()}
        ltl = id2ltl[ltl_id]
        # print("policy for ltl: ", ltl)

        # run rollouts
        edge2hits = rollout(tester, policy_bank, ltl, init_state, n_rollouts, max_depth)
    # save rollout results
    saver.save_worker_results(run_id, ltl_id, init_state, edge2hits, n_rollouts)


def rollout(tester, policy_bank, ltl, init_loc, n_rollouts, max_depth):
    """
    Rollout trained policy from init_loc to see which outgoing edges it satisfies
    """
    edge2hits = defaultdict(int)
    task_aux = Game(tester.get_task_params(policy_bank.policies[policy_bank.get_id(ltl)].f_task, ltl))
    initial_state = task_aux.dfa.state  # get default DFA initial state before progressing on agent's init_loc
    for rollout in range(n_rollouts):
        # print("\nrollout:", rollout)
        # print("init_loc: ", init_loc)
        # print("initial_state: ", initial_state)

        # Overwrite default agent start location and DFA initial state
        task = Game(tester.get_task_params(policy_bank.policies[policy_bank.get_id(ltl)].f_task, ltl, init_loc))
        # print("cur DFA state: ", task.dfa.state)
        # print("ltl: ", ltl)
        # print("full ltl: ", policy_bank.policies[policy_bank.get_id(ltl)].f_task)

        traversed_edge = None
        if initial_state != task.dfa.state:  # agent starts at a loc that already triggers a desired transition
            traversed_edge = task.dfa.nodelist[initial_state][task.dfa.state]
            # print("traversed edge before while: ", traversed_edge)
        depth = 0
        while not traversed_edge and not task.ltl_game_over and not task.env_game_over and depth <= max_depth:
            s1 = task.get_features()
            action = Actions(policy_bank.get_best_action(ltl, s1.reshape((1, len(task.get_features())))))
            prev_state = task.dfa.state
            _ = task.execute_action(action)
            # print(prev_state, action, task.dfa.state)
            if prev_state != task.dfa.state:
                traversed_edge = task.dfa.nodelist[prev_state][task.dfa.state]
                # print("traversed edge  in while: ", traversed_edge)
            depth += 1
        if traversed_edge:
            if traversed_edge not in policy_bank.policies[policy_bank.get_id(ltl)].get_edge_labels():
                print("ERROR: policy %s traversed invalid outgoing edge %s from location %s" % (str(ltl), str(traversed_edge), str(init_loc)))
            else:
                edge2hits[traversed_edge] += 1
    # print(edge2hits)
    return edge2hits


if __name__ == "__main__":
    algos = ["dqn-l", "hrl-e", "hrl-l", "lpopl", "zero_shot_transfer"]
    id2tasks = {
        0: "sequence",
        1: "interleaving",
        2: "safety",
        3: "transfer_sequence",
        4: "transfer_interleaving"
    }  # for reference

    parser = argparse.ArgumentParser(prog="run_single_rollout", description='Rollout a trained policy from a given state.')
    parser.add_argument('--algo', default='lpopl', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algos))
    parser.add_argument('--tasks_id', default=4, type=int,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(id2tasks.keys()))
    parser.add_argument('--map_id', default=0, type=int,
                        help='This parameter indicated the ID of map to run rollouts')
    parser.add_argument('--run_id', default=0, type=int,
                        help='This parameter indicated the ID of the training run when models are saved')
    parser.add_argument('--ltl_id', default=9, type=int,
                        help='This parameter indicated the ID of trained policy to rollout')
    parser.add_argument('--state_id', default=180, type=int,
                        help='This parameter indicated the ID of state in which rollouts start')
    parser.add_argument('--n_rollouts', default=100, type=int,
                        help='This parameter indicated the number of rollouts')
    parser.add_argument('--max_depth', default=100, type=int,
                        help='This parameter indicated maximum depth of a rollout')
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.tasks_id not in id2tasks: raise NotImplementedError("Tasks " + str(id2tasks[args.tasks_id]) + " hasn't been defined yet")
    if not(-1 <= args.map_id < 10): raise NotImplementedError("The map must be a number between -1 and 9")

    classifier_dpath = os.path.join("../tmp/", "task_%d/map_%d" % (args.tasks_id, args.map_id), "classifier")

    single_worker_rollouts(args.algo, classifier_dpath,
                           args.run_id, args.ltl_id, args.state_id, args.n_rollouts, args.max_depth)
