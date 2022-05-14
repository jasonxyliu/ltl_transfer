import time
import argparse
from game import GameParams, Game
from dataset_creator import read_train_test_formulas
from value_iteration import evaluate_optimal_policy

if __name__ == "__main__":
    # EXAMPLE: python compute_optimal_values.py --map=0 --train_type=soft_strict --train_size=50

    # Getting params
    train_types = [
        "sequence",
        "interleaving",
        "safety",
        "transfer_sequence",
        "transfer_interleaving",
        "hard",
        "mixed",
        "soft_strict",
        "soft",
        "no_orders",
    ]

    parser = argparse.ArgumentParser(prog="compute_optimal_steps", description='Run value iteration to find optimal steps.')
    parser.add_argument('--map', default=-1, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
    parser.add_argument('--train_type', default='all', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=50, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_size', default=100, type=int,
                        help='This parameter indicated the number of LTLs in the test set')
    parser.add_argument('--dataset_name', default='spot', type=str, choices=['minecraft', 'spot'],
                        help='This parameter indicated the dataset to read tasks from')
    args = parser.parse_args()
    if args.train_type not in train_types+['all']: raise NotImplementedError("Training Tasks " + str(args.train_type) + " hasn't been defined yet")
    if not(-1 <= args.map < 21): raise NotImplementedError("The map must be a number between -1 and 21")

    map_ids = range(10) if args.map == -1 else [args.map]
    task_ids = [train_types.index(train_type) for train_type in train_types[5:]] if args.train_type == 'all' else [train_types.index(args.train_type)]
    train_size = args.train_size
    test_size = args.test_size
    consider_night = False

    for map_id in map_ids:
        map_fpath = "../experiments/maps/map_%d.txt" % map_id
        policy_fpath = "../experiments/optimal_policies/map_%d.txt" % map_id
        open(policy_fpath, 'a').close()
        for task_id in task_ids:
            # Retrieve tasks for a LTL type
            task_type = train_types[task_id]
            train_tasks, _ = read_train_test_formulas(dataset_name=args.dataset_name, train_set_type=task_type, train_size=train_size, test_size=test_size)
            task_aux = Game(GameParams(map_fpath, train_tasks[0], consider_night, init_dfa_state=None, init_loc=None))
            time_init = time.time()
            # Compute optimal steps for tasks of this LTL type in this map
            out_str = evaluate_optimal_policy(task_aux.map_array, task_aux.agent.i, task_aux.agent.j, consider_night, train_tasks, task_id+1, task_type)
            # Update the optimal policy file corresponding to the map_id with computed optimal steps
            with open(policy_fpath, "r") as rfile:
                lines = rfile.readlines()  # readlines reads the newline character at the end of a line
            while task_id >= len(lines):  # equal in case optimal policy file does not have newline EOF
                lines.append("\n")
            lines[task_id] = out_str
            with open(policy_fpath, "w") as wfile:
                wfile.writelines(lines)  # writelines does not write a newline character to the end of a line
            print("task %s, map_%d took: %0.2f mins\n" % (task_type, map_id, (time.time() - time_init)/60))
        print("\n")
