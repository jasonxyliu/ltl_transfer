import argparse
import baseline_dqn
import baseline_hrl
import lpopl
import zero_shot_transfer
from test_utils import TestingParameters, Tester, Saver, Loader
from curriculum import CurriculumLearner


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


def run_experiment(alg_name, map_id, tasks_id, train_type, train_size, test_type, num_times, r_good, run_id, relabel_method, show_print):
    # configuration of testing params
    testing_params = TestingParameters()

    # configuration of learning params
    learning_params = LearningParameters()

    # Setting the experiment
    tester = Tester(learning_params, testing_params, map_id, tasks_id, train_type, train_size, test_type)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.tasks, r_good=r_good)

    # Setting up the saver
    saver = Saver(alg_name, tester)
    loader = Loader(saver)

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
        lpopl.run_experiments(tester, curriculum, saver, num_times, show_print)

    # Relabel state-centric options learn by LPOPL then zero-shot transfer
    if alg_name == "zero_shot_transfer":
        zero_shot_transfer.run_experiments(tester, curriculum, saver, loader, run_id, relabel_method)


def run_multiple_experiments(alg, tasks_id, train_type, train_size, test_type, run_id, relabel_method):
    num_times = 3
    r_good     = 0.5 if tasks_id == 2 else 0.9
    show_print = True

    for map_id in range(10):
        print("Running", "r_good:", r_good, "alg:", alg, "map_id:", map_id, "tasks:", train_type)
        run_experiment(alg, map_id, tasks_id, train_type, train_size, test_type, num_times, r_good, run_id, relabel_method, show_print)


def run_single_experiment(alg, tasks_id, train_type, train_size, test_type, map_id, run_id, relabel_method):
    num_times  = 1  # each algo was run 3 times per map in the paper
    r_good     = 0.5 if tasks_id == 2 else 0.9
    show_print = True

    print("Running", "r_good:", r_good, "alg:", alg, "map_id:", map_id, "tasks:", train_type)
    run_experiment(alg, map_id, tasks_id, train_type, train_size, test_type, num_times, r_good, run_id, relabel_method, show_print)


if __name__ == "__main__":
    # EXAMPLE: python run_experiments.py --algo=lpopl --train_type=sequence --map=0

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

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    parser.add_argument('--algo', default='lpopl', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algos))
    parser.add_argument('--train_type', default='sequence', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=10, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_type', default='sequence', type=str,
                        help='This parameter indicated which test tasks to solve. The options are: ' + str(test_types))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
    parser.add_argument('--run_id', default=0, type=int,
                        help='This parameter indicated the policy bank saved after which run will be used for transfer')
    # parser.add_argument('--load_trained', action="store_true",
    #                     help='This parameter indicated whether to load trained policy models. Include it in command line to load trained policies')
    parser.add_argument('--relabel_method', default='cluster', type=str,
                        help='This parameter indicated which method is used to relabel state-centric options. The options are: ' + str(relabel_methods))
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.train_type not in train_types: raise NotImplementedError("Training Tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError("Test Tasks " + str(args.test_type) + " hasn't been defined yet")
    if not(-1 <= args.map < 10): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    alg        = args.algo
    tasks_id   = train_types.index(args.train_type)
    train_type = args.train_type
    train_size = args.train_size
    test_type  = args.test_type
    map_id     = args.map

    if map_id > -1:
        run_single_experiment(alg, tasks_id, train_type, train_size, test_type, map_id, args.run_id, args.relabel_method)
    else:
        run_multiple_experiments(alg, tasks_id, train_type, train_size, test_type, args.run_id, args.relabel_method)
