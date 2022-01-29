import os
import dill
import argparse
from collections import defaultdict
from game import *


def rollout(task_aux, policy, init_state, n_rollouts, max_depth):
    """
    Rollout trained policy from init_state to see which outgoing edge it satisfies
    """
    edge2hits = defaultdict(int)
    for _ in range(n_rollouts):
        depth = 0
        traversed_edge = None
        task_aux.set_agent_loc(init_state)
        while depth <= max_depth:
            s1 = task_aux.get_features()
            action = Actions(policy.sess.run(policy.get_best_action(), {policy.s1: s1}))
            prev_state = task_aux.dfa.state
            reward = task_aux.execute_action(action)
            if reward == 1:
                traversed_edge = task_aux.dfa.nodelist[prev_state][task_aux.dfa.state]
                break
            if task_aux.ltl_game_over or task_aux.env_game_over:
                break
            depth += 1
        if traversed_edge:
            edge2hits[traversed_edge] += 1
    max_edge = None
    if edge2hits:
        max_edge = max(edge2hits.items(), key=lambda kv: kv[1])[0]
    return max_edge


def load_policy_bank(policy_id):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_single_rollout", description='Rollout a trained policy from a given state.')
    parser.add_argument('--map_id', default=0, type=int,
                        help='This parameter indicated the ID of map to run rollouts')
    parser.add_argument('--policy_id', default=0, type=int,
                        help='This parameter indicated the ID of trained policy to rollout')
    parser.add_argument('--init_state_id', default=0, type=int,
                        help='This parameter indicated the ID of state in which rollouts start')
    parser.add_argument('--n_rollouts', default=100, type=int,
                        help='This parameter indicated the number of rollouts')
    parser.add_argument('--max_depth', default=100, type=int,
                        help='This parameter indicated maximum depth of a rollout')
    args = parser.parse_args()

    classifier_dname = os.path.join("results", "classifier", args.map_id)

    with open(os.path.join(classifier_dname, "tester.pkl"), "rb") as file:
        tester = dill.load(file)
    task_aux = Game(tester.get_task_params(tester.get_transfer_tasks()[0]))

    policy = load_policy(os.path.join(classifier_dname, args.policy_id))

    with open(os.path.join(classifier_dname, "states.pkl"), "rb") as file:
        id2state = dill.load(file)
    init_state = id2state[args.init_state_id]

    max_edge = rollout(task_aux, policy, init_state, args.n_rollouts, args.max_depth)

    with open(os.path.join(classifier_dname, "results.txt"), "a") as file:
        line = " "
        if max_edge:
            line = "%d %d %s\n" % (args.policy_id, args.init_state_id, max_edge)
        file.write(line)
