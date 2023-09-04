import sys
import argparse
import baseline_dqn
import baseline_hrl
import lpopl
import zero_shot_transfer_spot
from test_utils import TestingParameters, Tester, Saver
from curriculum import CurriculumLearner

import bosdyn.client
import bosdyn.client.util


class LearningParameters:
    def __init__(self, lr=0.0001, max_timesteps_per_task=50000, buffer_size=25000,
                print_freq=1000, exploration_fraction=0.1, exploration_final_eps=0.02,
                train_freq=1, batch_size=32, learning_starts=1000, gamma=0.9,
                target_network_update_freq=100):
        """Parameters
        -------
        lr: float
            learning rate for adam optimizer
        max_timesteps_per_task: int
            number of env steps to optimizer for per task
        buffer_size: int
            size of the replay buffer
        print_freq: int
            how often to print out training progress
            set to None to disable printing
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
            set to None to disable printing
        batch_size: int
            size of a batched sampled from replay buffer for training
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        gamma: float
            discount factor
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        """
        self.lr = lr
        self.max_timesteps_per_task = max_timesteps_per_task
        self.buffer_size = buffer_size
        self.print_freq = print_freq
        self.exploration_fraction = exploration_fraction
        self.exploration_final_eps = exploration_final_eps
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.target_network_update_freq = target_network_update_freq


def run_experiment(args, tasks_id, num_times, r_good, show_print):
    alg_name = args.algo
    map_id = args.map
    dataset_name = args.dataset_name
    train_type = args.train_type
    train_size = args.train_size
    test_type = args.test_type
    total_steps = args.total_steps
    increment_steps = args.incremental_steps
    run_id = args.run_id,
    relabel_method = args.relabel_method
    transfer_num_times = args.transfer_num_times
    edge_matcher = args.edge_matcher

    # configuration of testing params
    testing_params = TestingParameters()

    # configuration of learning params
    learning_params = LearningParameters()

    # Setting the experiment
    tester = Tester(learning_params, testing_params, map_id, tasks_id, dataset_name, train_type, train_size, test_type, edge_matcher)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.tasks, r_good=r_good, total_steps=total_steps)

    # Setting up the saver
    saver = Saver(alg_name, tester)

    # Baseline 1 (standard DQN with Michael Littman's approach)
    if alg_name == "dqn-l":
        baseline_dqn.run_experiments(tester, curriculum, saver, num_times, show_print)

    # Baseline 2 (Hierarchical RL)
    if alg_name == "hrl-e":
        baseline_hrl.run_experiments(tester, curriculum, saver, num_times, show_print, use_dfa=False)

    # Baseline 3 (Hierarchical RL with LTL constraints)
    if alg_name == "hrl-l":
        baseline_hrl.run_experiments(tester, curriculum, saver, num_times, show_print, use_dfa=True)

    # LPOPL
    if alg_name == "lpopl":
        lpopl.run_experiments(tester, curriculum, saver, num_times, increment_steps, show_print)

    # Relabel state-centric options learn by LPOPL then zero-shot transfer
    if alg_name == "zero_shot_transfer":
        zero_shot_transfer_spot.run_experiments(tester, curriculum, saver, run_id, relabel_method, transfer_num_times, args)


def run_multiple_experiments(args, tasks_id):
    num_times = 3
    r_good    = 0.5 if tasks_id == 2 else 0.9
    show_print = True

    for map_id in range(10):
        print("Running r_good: %0.2f; domain: %s; alg: %s; map_id: %d; train_type: %s; train_size: %d; test_type: %s; edge_mather: %s" % (r_good, args.dataset_name, args.algo, map_id, args.train_type, args.train_size, args.test_type, args.edge_matcher))
        run_experiment(args, tasks_id, num_times, r_good, show_print)


def run_single_experiment(args, tasks_id):
    num_times = 1  # each algo was run 3 times per map in the paper
    r_good    = 0.5 if tasks_id == 2 else 0.9
    show_print = True

    print("Running r_good: %0.2f; domain: %s; alg: %s; map_id: %d; train_type: %s; train_size: %d; test_type: %s; edge_mather: %s" % (r_good, args.dataset_name, args.algo, args.map, args.train_type, args.train_size, args.test_type, args.edge_matcher))
    run_experiment(args, tasks_id, num_times, r_good, show_print)


def main(argv):
    # Getting params
    algos = ["dqn-l", "hrl-e", "hrl-l", "lpopl", "zero_shot_transfer"]
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
    relabel_methods = ["cluster", "parallel"]

    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--algo', default='lpopl', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algos))
    parser.add_argument('--train_type', default='hard', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=2, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_type', default='hard', type=str,
                        help='This parameter indicated which test tasks to solve. The options are: ' + str(test_types))
    parser.add_argument('--map', default=20, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
    parser.add_argument('--total_steps', default=10000, type=int,
                        help='This parameter indicated the increment to the total training steps')
    parser.add_argument('--incremental_steps', default=0, type=int,
                        help='This parameter indicated the increment to the total training steps')
    parser.add_argument('--run_id', default=0, type=int,
                        help='This parameter indicated the policy bank saved after which run will be used for transfer')
    # parser.add_argument('--load_trained', action="store_true",
    #                     help='This parameter indicated whether to load trained policy models. Include it in command line to load trained policies')
    parser.add_argument('--relabel_method', default='parallel', type=str, choices=['cluster', 'parallel'],
                        help='This parameter indicated which method is used to relabel state-centric options. The options are: ' + str(
                            relabel_methods))
    parser.add_argument('--transfer_num_times', default=1, type=int,
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--edge_matcher', default='relaxed', type=str, choices=['rigid', 'relaxed'],
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--dataset_name', default='spot', type=str, choices=['minecraft', 'spot'],
                        help='This parameter indicated the dataset to read tasks from')
    # Spot config
    parser.add_argument("--username", type=str, default="user", help="Username of Spot")
    parser.add_argument("--password", type=str, default="97qp5bwpwf2c", help="Password of Spot")  # dungnydsc8su
    parser.add_argument("--dock_id", required=True, type=int, help="Docking station ID to dock at")
    parser.add_argument("--time_per_move", type=int, default=25, help="Seconds each move in grid should take")
    parser.add_argument('--dock_after_use', action="store_true", help='Include to dock Spot after operation')
    parser.add_argument('--poweroff_after_use', action="store_true", help='Include to power off Spot after operation')

    args = parser.parse_args(argv)
    bosdyn.client.util.setup_logging(args.verbose)
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.train_type not in train_types: raise NotImplementedError(
        "Training tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError(
        "Test tasks " + str(args.test_type) + " hasn't been defined yet")
    if not (-1 <= args.map < 21): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    tasks_id = train_types.index(args.train_type)
    if args.map > -1:
        run_single_experiment(args, tasks_id)
    else:
        run_multiple_experiments(args, tasks_id)


if __name__ == "__main__":
    # EXAMPLE: python run_experiments_spot.py 138.16.161.12 --algo=zero_shot_transfer --dock_id=521
    if not main(sys.argv[1:]):
        sys.exit(-1)