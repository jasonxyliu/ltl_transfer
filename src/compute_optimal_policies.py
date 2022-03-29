import time
import argparse
from game import GameParams, Game
from dataset_creator import read_test_train_formulas
from value_iteration import evaluate_optimal_policy

if __name__ == "__main__":
    # EXAMPLE: python compute_optimal_policies.py  --map=0 --train_type=soft_strict --train_size=50 --test_type=soft_strict

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
    test_types = [
        "hard",
        "mixed",
        "soft_strict",
        "soft",
        "no_orders",
    ]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    parser.add_argument('--map', default=-1, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
    parser.add_argument('--train_type', default='all', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=50, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_type', default='soft_strict', type=str,
                        help='This parameter indicated which test tasks to solve. The options are: ' + str(test_types))
    args = parser.parse_args()
    if args.train_type not in train_types+['all']: raise NotImplementedError("Training Tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError("Test Tasks " + str(args.test_type) + " hasn't been defined yet")
    if not(-1 <= args.map < 10): raise NotImplementedError("The map must be a number between -1 and 9")

    map_ids = range(10) if args.map == -1 else [args.map]
    task_ids = [train_types.index(train_type) for train_type in test_types] if args.train_type == 'all' else [train_types.index(args.train_type)]
    train_size = args.train_size
    test_type = args.test_type
    consider_night = False

    for map_id in map_ids:
        map_fpath = "../experiments/maps/map_%d.txt" % map_id
        for task_id in task_ids:
            task_type = train_types[task_id]
            train_tasks, _ = read_test_train_formulas(task_type, test_type, train_size)
            task_aux = Game(GameParams(map_fpath, train_tasks[0], consider_night, init_dfa_state=None, init_loc=None))
            time_init = time.time()
            evaluate_optimal_policy(task_aux.map_array, task_aux.agent.i, task_aux.agent.j, consider_night, train_tasks, task_id+1)
            print("task %s, map_%d took: %0.2f mins\n" % (task_type, map_id, (time.time() - time_init)/60))
        print("\n")
