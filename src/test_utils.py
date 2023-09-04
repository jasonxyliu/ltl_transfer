import os
import json
import dill
import argparse
import logging
from collections import defaultdict
import numpy as np
try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
from game import GameParams, Game
from dataset_creator import read_train_test_formulas
import tasks


class TestingParameters:
    def __init__(self, test=True, test_freq=1000, num_steps=100):
        """Parameters
        -------
        test: bool
            if True, we test current policy during training
        test_freq: int
            test the model every `test_freq` steps.
        num_steps: int
            number of steps during testing
        """
        self.test = test
        self.test_freq = test_freq
        self.num_steps = num_steps


def _get_optimal_values(file, experiment):
    f = open(file)
    lines = [line.rstrip() for line in f]
    f.close()
    return eval(lines[experiment])


class Tester:
    def __init__(self, learning_params, testing_params, map_id, prob, tasks_id, dataset_name, train_type, train_size, test_type, edge_matcher, save_dpath, file_results=None):
        if file_results is None:
            # setting the test attributes
            self.learning_params = learning_params
            self.testing_params = testing_params
            self.map_id = map_id
            self.transition_type = "deterministic" if prob == 1.0 else "stochastic"
            self.prob = prob
            self.tasks_id = tasks_id
            self.dataset_name = dataset_name
            self.train_type = train_type
            self.train_size = train_size
            self.test_type = test_type
            self.edge_matcher = edge_matcher
            self.save_dpath = save_dpath
            self.experiment = f"{train_type}/map_{map_id}"
            self.map = f"{save_dpath}/experiments/maps/map_{map_id}.txt"
            self.consider_night = False
            if dataset_name == "spot":
                self.experiment = f"spot/{train_type}/map_{map_id}"
                self.experiment_train = f"spot/{train_type}_2/map_{map_id}"
                train_tasks, self.transfer_tasks = read_train_test_formulas(dataset_name, train_type, test_type, 2, 4)
                self.tasks = train_tasks[0:train_size]
                self.transfer_results_dpath = os.path.join("../results/spot", f"{train_type}_{train_size}_{test_type}_{edge_matcher}", f"map_{map_id}")
            elif train_type == "sequence":
                self.tasks = tasks.get_sequence_of_subtasks()
            elif train_type == "interleaving":
                self.tasks = tasks.get_interleaving_subtasks()
            elif train_type == "safety":
                self.tasks = tasks.get_safety_constraints()
                self.consider_night = True
            elif train_type == 'random':
                self.experiment = f"{train_type}/map_{map_id}"
                self.experiment_train = f"{train_type}/map_{map_id}"
                train_tasks, self.transfer_tasks = read_train_test_formulas(dataset_name, 'hard', test_type, 50)
                self.tasks = train_tasks[0: train_size]
                self.transfer_results_dpath = os.path.join("..", "results_test", f"{train_type}_{test_type}_{edge_matcher}", f"map_{map_id}")
                os.makedirs(self.transfer_results_dpath, exist_ok=True)
                self.transfer_log_fpath = os.path.join(self.transfer_results_dpath, "zero_shot_transfer_log.txt")
                logging.basicConfig(filename=self.transfer_log_fpath, filemode='w', level=logging.INFO, format="%(message)s")
            else:  # transfer tasks
                if train_type == 'transfer_sequence':
                    self.tasks = tasks.get_sequence_training_tasks()
                    self.transfer_tasks = tasks.get_transfer_tasks()
                    self.transfer_results_dpath = os.path.join(save_dpath, "results", train_type, f"map_{map_id}")
                elif train_type == 'transfer_interleaving':
                    self.tasks = tasks.get_interleaving_training_tasks()
                    self.transfer_tasks = tasks.get_transfer_tasks()
                    self.transfer_results_dpath = os.path.join(save_dpath, "results", train_type, f"map_{map_id}")
                else:
                    self.experiment = f"{train_type}_{train_size}/map_{map_id}/prob_{self.prob}"
                    self.experiment_train = f"{train_type}_50/map_{map_id}/prob_{self.prob}"
                    train_tasks, self.transfer_tasks = read_train_test_formulas(save_dpath, dataset_name, train_type, test_type, 50)
                    self.tasks = train_tasks[0: train_size]
                    self.transfer_results_dpath = os.path.join(save_dpath, "results_icra24", self.transition_type, f"{train_type}_{train_size}_{test_type}_{edge_matcher}", f"map_{map_id}", f"prob_{self.prob}")
                os.makedirs(self.transfer_results_dpath, exist_ok=True)
                self.transfer_log_fpath = os.path.join(self.transfer_results_dpath, "zero_shot_transfer_log.txt")
                logging.basicConfig(filename=self.transfer_log_fpath, filemode='w', level=logging.INFO, format="%(message)s")

            # load pre-computed optimal steps for 'task_type' in 'map_id'
            optimal_aux = _get_optimal_values(f'{save_dpath}/experiments/optimal_policies/map_{map_id}.txt', tasks_id)

            # I store the results here
            self.results = {}
            self.optimal = {}
            self.steps = []
            for idx, task in enumerate(self.tasks):
                self.optimal[task] = learning_params.gamma ** (float(optimal_aux[idx]) - 1)
                self.results[task] = {}
            # save results for transfer learning
            if tasks_id > 2:
                self.task2run2sol = {str(transfer_task): defaultdict(list) for transfer_task in self.transfer_tasks}
                self.task2run2trajs = {str(transfer_task): defaultdict(list) for transfer_task in self.transfer_tasks}
                self.task2success = {str(transfer_task): 0.0 for transfer_task in self.transfer_tasks}
        else:
            # Loading precomputed results
            data = read_json(file_results)
            self.results = dict([(eval(t), data["results"][t]) for t in data["results"]])
            self.optimal = dict([(eval(t), data["optimal"][t]) for t in data["optimal"]])
            self.steps = data["steps"]
            self.tasks = [eval(t) for t in data["tasks"]]
            # obs: json transform the integer keys from 'results' into strings
            # so I'm changing the 'steps' to strings
            for idx, step in enumerate(self.steps):
                self.steps[idx] = str(step)

    def get_LTL_tasks(self):
        return self.tasks

    def get_transfer_tasks(self):
        return self.transfer_tasks

    def get_task_params(self, ltl_task, init_dfa_state=None, init_loc=None):
        return GameParams(self.map, self.prob, ltl_task, self.consider_night, init_dfa_state, init_loc)

    def run_test(self, step, sess, test_function, *test_args):
        # 'test_function' parameters should be (sess, task_params, learning_params, testing_params, *test_args)
        # and returns the reward
        for task in self.tasks:
            task_params = self.get_task_params(task)
            reward = test_function(sess, task_params, self.learning_params, self.testing_params, *test_args)
            if step not in self.results[task]:
                self.results[task][step] = []  # store reward per run for a total of 'num_times' runs
            if len(self.steps) == 0 or self.steps[-1] < step:
                self.steps.append(step)
            self.results[task][step].append(reward)

    def show_results(self):
        # Computing average performance per task
        average_reward = {}
        for t in self.tasks:
            for s in self.steps:
                normalized_rewards = [r / self.optimal[t] for r in self.results[t][s]]
                a = np.array(normalized_rewards)
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]
        # Showing average performance across all the task
        print("\nAverage discounted reward on this map --------------------")
        print("\tsteps\tP25\tP50\tP75")
        num_tasks = float(len(self.tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            p25, p50, p75 = get_precentiles_str(normalized_rewards)
            print("\t" + str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)

    def export_results(self):
        # Showing performance per task
        average_reward = {}
        for t in self.tasks:
            for s in self.steps:
                normalized_rewards = [r / self.optimal[t] for r in self.results[t][s]]
                a = np.array(normalized_rewards)
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]
        # Computing average performance across all the task
        ret = []
        num_tasks = float(len(self.tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            ret.append([s, normalized_rewards])
        return ret

    def log_results(self, data, log=True):
        if log:
            logging.info(data)
        else:
            print(data)


class Saver:
    def __init__(self, alg_name, tester):
        self.alg_name = alg_name
        self.tester = tester

        self.exp_dir = os.path.join(tester.save_dpath, "options", tester.transition_type, tester.experiment_train)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.train_dpath = os.path.join(self.exp_dir, "train_data")
        os.makedirs(self.train_dpath, exist_ok=True)

        self.policy_dpath = os.path.join(self.train_dpath, "policy_model")
        os.makedirs(self.policy_dpath, exist_ok=True)

        self.classifier_dpath = os.path.join(self.exp_dir, "classifier")
        os.makedirs(self.classifier_dpath, exist_ok=True)

        self.file_out = os.path.join(self.exp_dir, alg_name + ".json")  # tasks_id>=3, store training results for transfer

    def save_train_data(self, curriculum, run_id):
        run_dpath = os.path.join(self.train_dpath, f"run_{run_id}")
        os.makedirs(run_dpath, exist_ok=True)
        # save tester
        save_pkl(os.path.join(run_dpath, "tester.pkl"), self.tester)
        # save curriculum
        save_pkl(os.path.join(run_dpath, "curriculum.pkl"), curriculum)

    def save_policy_bank(self, policy_bank, run_id):
        tf_saver = tf.train.Saver()
        policy_bank_prefix = os.path.join(self.policy_dpath, f"run_{run_id}", "policy_bank")
        tf_saver.save(policy_bank.sess, policy_bank_prefix)

    def save_results(self):
        results = {
            "tasks": [str(t) for t in self.tester.tasks],
            "optimal": dict([(str(t), self.tester.optimal[t]) for t in self.tester.optimal]),
            "steps": self.tester.steps,
            "results": dict([(str(t), self.tester.results[t]) for t in self.tester.results])
        }
        save_json(self.file_out, results)

    def save_transfer_data(self, task_aux, id2ltls):
        """
        Save all data needed to learn classifiers in parallel
        """
        # save tester
        save_pkl(os.path.join(self.classifier_dpath, "tester.pkl"), self.tester)
        # save valid agent locations where rollouts start
        id2state = {}
        state2id = {}
        for x in range(task_aux.map_height):
            for y in range(task_aux.map_width):
                if task_aux.is_valid_agent_loc(x, y):
                    id2state[len(id2state)] = (x, y)
                    state2id[(x, y)] = len(state2id)
        save_pkl(os.path.join(self.classifier_dpath, "states.pkl"), id2state)
        save_json(os.path.join(self.classifier_dpath, "states.json"), id2state)
        # save map from subtask ltl_id to (subtask ltl, its corresponding full ltl)
        save_pkl(os.path.join(self.classifier_dpath, "id2ltls.pkl"), id2ltls)
        save_json(os.path.join(self.classifier_dpath, "id2ltls.json"), id2ltls)
        return state2id

    def save_transfer_results(self):
        results = {
            'transfer_tasks': [str(t) for t in self.tester.transfer_tasks],
            'task2run2sol': self.tester.task2run2sol,
            'task2success': self.tester.task2success,
            'task2run2trajs': self.tester.task2run2trajs
        }
        transfer_results_fpath = os.path.join(self.tester.transfer_results_dpath, "zero_shot_transfer_results.json")
        save_json(transfer_results_fpath, results)

    def save_worker_results(self, run_idx, ltl_id, state, edge2hits, n_rollouts):
        """
        Save results from a worker
        """
        rollout_results = {
            "run_idx": run_idx,
            "ltl": ltl_id,
            "state": state,
            "edge2hits": edge2hits,
            "n_rollouts": n_rollouts,
        }
        worker_fpath = os.path.join(self.classifier_dpath, f"ltl{ltl_id}_state{state[0]}-{state[1]}_")
        save_pkl(worker_fpath+"rollout_results_parallel.pkl", rollout_results)
        save_json(worker_fpath + "rollout_results_parallel.json", rollout_results)

    def save_rollout_results(self, fname, policy2loc2edge2hits_pkl, policy2loc2edge2hits_json):
        """
        Save results of rolling out state-centric policies from various states to compute initiation set classifiers
        """
        save_pkl(os.path.join(self.classifier_dpath, fname+".pkl"), policy2loc2edge2hits_pkl)
        save_json(os.path.join(self.classifier_dpath, fname+".json"), policy2loc2edge2hits_json)


class Loader:
    def __init__(self, saver):
        self.saver = saver

    def load_policy_bank(self, run_idx, sess):
        run_dpath = os.path.join(self.saver.policy_dpath, f"run_{run_idx}")  # where all tf model are saved
        # saver = tf.train.import_meta_graph(run_dpath+"policy_bank.meta")
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(run_dpath))


def get_precentiles_str(a):
    p25 = "%0.2f" % float(np.percentile(a, 25))
    p50 = "%0.2f" % float(np.percentile(a, 50))
    p75 = "%0.2f" % float(np.percentile(a, 75))
    return p25, p50, p75


def transfer_metrics(train_type, train_size, test_type, map_id, prob, num_times, edge_matcher, save_dpath):
    """
    Compute evaluation metrics for zero-shot transfer
    """
    results_dpath = os.path.join(save_dpath, "results_icra24", f"{train_type}_{train_size}_{test_type}_{edge_matcher}", f"map_{map_id}", f"prob_{prob}")
    results = read_json(os.path.join(results_dpath, "zero_shot_transfer_results.json"))
    task2success = results["task2success"]
    success_rates = []
    num_success = 0

    num_tasks = 0
    for task, success_rate in task2success.items():
        success_rates.append(success_rate)
        num_success += success_rate * num_times
        num_tasks += 1

    p25, p50, p75 = get_precentiles_str(success_rates)
    mean, std = np.mean(success_rates), np.std(success_rates)

    metrics_fpath = os.path.join(results_dpath, "zero_shot_transfer_metrics.txt")
    with open(metrics_fpath, "w") as wfile:
        wfile.write(f"{train_type}_{train_size}_{test_type}\n")
        wfile.write("%0.2f += %0.2f\n" % (mean, std))
        wfile.write(p25 + "\t" + p50 + "\t" + p75 + "\n")
        wfile.write(f"total number of successes in {num_times} runs: {num_success}\n")  # some test types may not have 100 unique tasks
        wfile.write(f"number of unique test tasks in test type {test_type}: {num_tasks}")


def export_results(algorithm, task_type, prob, save_dpath):
    transition_type = "stochastic" if prob == 1.0 else "deterministic"
    for map_type, maps in [("random", range(0, 5)), ("adversarial", range(5, 10))]:
        # Computing the summary of the results
        normalized_rewards = None
        for map_id in maps:
            result = f"{save_dpath}/options/{transition_type}/{task_type}/map_{map_id}/prob_{prob}/{algorithm}.json"
            tester = Tester(None, None, None, None, None, None, None, None, None, None, result)
            ret = tester.export_results()
            if normalized_rewards is None:
                normalized_rewards = ret
            else:
                for j in range(len(normalized_rewards)):
                    normalized_rewards[j][1] = np.append(normalized_rewards[j][1], ret[j][1])
        # Saving the results
        folders_out = f"{save_dpath}/results_icra24/{transition_type}/{task_type}/{map_type}/prob_{prob}"
        if not os.path.exists(folders_out): os.makedirs(folders_out)
        file_out = f"{folders_out}/{algorithm}.txt"
        f_out = open(file_out, "w")
        for step in range(len(normalized_rewards)):
            print(normalized_rewards[step][0], normalized_rewards[step][1])
            p25, p50, p75 = get_precentiles_str(normalized_rewards[step][1])
            f_out.write(str(normalized_rewards[step][0]) + "\t" + p25 + "\t" + p50 + "\t" + p75 + "\n")
        f_out.close()


def save_pkl(fpath, data):
    with open(fpath, "wb") as file:
        dill.dump(data, file)


def load_pkl(fpath):
    with open(fpath, 'rb') as file:
        data = dill.load(file)
    return data


def save_json(fpath, data):
    with open(fpath, 'w') as outfile:
        json.dump(data, outfile)


def read_json(fpath):
    with open(fpath) as data_file:
        data = json.load(data_file)
    return data


def aggregate_transfer_results(results_dpath, num_tasks):
    fnames = os.listdir(results_dpath)

    for task_id in range(num_tasks):
        if f"test_ltl_{task_id}.txt" not in fnames:
            print(task_id)


if __name__ == "__main__":
    # EXAMPLE for export training results: python test_utils.py --algo=lpopl --train_type=sequence
    # EXAMPLE for export transfer results: python test_utils.py --train_type=no_orders --train_size=50 --test_type=hard --map=0 --transfer_num_times=1

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

    parser = argparse.ArgumentParser(prog="run_experiments", description="Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.")
    parser.add_argument("--algo", default="lpopl", type=str,
                        help="This parameter indicated which RL algorithm to use. The options are: " + str(algos))
    parser.add_argument('--train_type', default='no_orders', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(train_types))
    parser.add_argument('--train_size', default=50, type=int,
                        help='This parameter indicated the number of LTLs in the training set')
    parser.add_argument('--test_type', default='soft', type=str,
                        help='This parameter indicated which test tasks to solve. The options are: ' + str(test_types))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
    parser.add_argument('--prob', default=1.0, type=float,
                        help='transition stochasticity')
    parser.add_argument('--transfer_num_times', default=1, type=int,
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--edge_matcher', default='rigid', type=str, choices=['rigid', 'relaxed'],
                        help='This parameter indicated the number of times to run a transfer experiment')
    parser.add_argument('--save_dpath', default='..', type=str,
                        help='path to directory to save')
    args = parser.parse_args()
    if args.algo not in algos: raise NotImplementedError("Algorithm " + str(args.algo) + " hasn't been implemented yet")
    if args.train_type not in train_types: raise NotImplementedError(
        "Training tasks " + str(args.train_type) + " hasn't been defined yet")
    if args.test_type not in test_types: raise NotImplementedError(
        "Test tasks " + str(args.test_type) + " hasn't been defined yet")
    if not (-1 <= args.map < 10): raise NotImplementedError("The map must be a number between -1 and 9")

    # export_results(args.algo, args.train_type, self.prob, args.save_dpath)
    transfer_metrics(args.train_type, args.train_size, args.test_type, args.map, args.prob, args.transfer_num_times, args.edge_matcher, args.save_dpath)
