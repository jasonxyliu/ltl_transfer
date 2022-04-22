import os
import dill
import random
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from game import Game
from curriculum import CurriculumLearner
from lpopl import _initialize_policy_bank
from zero_shot_transfer import construct_initiation_set_classifiers, get_training_edges, dfa2graph, remove_infeasible_edges

classifier_dpath = os.path.join("../tmp/no_orders_50_duplicate/map_0/classifier")

debug_dpath = os.path.join("../tmp/no_orders_50_duplicate/map_0/", "debug")
os.makedirs(debug_dpath, exist_ok=True)

with open(os.path.join(classifier_dpath, "tester.pkl"), "rb") as file:
    tester = dill.load(file)

curriculum = CurriculumLearner(tester.tasks, r_good=0.9)
curriculum.restart()

random.seed(0)
sess = tf.Session()

policy_bank = _initialize_policy_bank(sess, tester.learning_params, curriculum, tester, load_tf=False)

policy2edge2loc2prob = construct_initiation_set_classifiers(classifier_dpath, policy_bank)

train_edges, edge2ltls = get_training_edges(policy_bank, policy2edge2loc2prob)

for task_idx, transfer_task in enumerate(tester.transfer_tasks):
    print("== Transfer Task %d: %s\n" % (task_idx, str(transfer_task)))

    task_aux = Game(tester.get_task_params(transfer_task))  # same grid map as the training tasks
    # Wrapper: DFA -> NetworkX graph
    dfa_graph = dfa2graph(task_aux.dfa)
    for line in nx.generate_edgelist(dfa_graph):
        print(line)

    # Remove edges in DFA that do not have a matching train edge
    test2trains = remove_infeasible_edges(dfa_graph, train_edges, task_aux.dfa.state, task_aux.dfa.terminal[0])
    print("\nNew DFA graph")
    for line in nx.generate_edgelist(dfa_graph):
        print(line)

    # Graph search to find all simple/feasible paths from initial state to goal state
    feasible_paths_node = list(nx.all_simple_paths(dfa_graph, source=task_aux.dfa.state, target=task_aux.dfa.terminal))
    feasible_paths_edge = [list(path) for path in map(nx.utils.pairwise, feasible_paths_node)]
    print("\ndfa start: %d; goal: %s" % (task_aux.dfa.state, str(task_aux.dfa.terminal)))
    print("feasible paths: %s\n" % str(feasible_paths_node))

    if not feasible_paths_node:
        plt.figure(figsize=(12, 12))
        pos = nx.circular_layout(dfa_graph)
        nx.draw_networkx(dfa_graph, pos, with_labels=True)
        nx.draw_networkx_edges(dfa_graph, pos)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.savefig(os.path.join(debug_dpath, "transfer_task_%d" % task_idx))
        plt.close()
        # plt.show()
