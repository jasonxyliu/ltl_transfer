import numpy as np
from game import GameParams
import os, json, tasks, argparse


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
    lines = [l.rstrip() for l in f]
    f.close()
    return eval(lines[experiment])


class Tester:
    def __init__(self, learning_params, testing_params, map_id, tasks_id, file_results=None):
        if file_results is None:
            # setting the test attributes
            self.experiment = "task_%d/map_%d"%(tasks_id,map_id)
            self.learning_params = learning_params
            self.testing_params = testing_params
            self.map     = '../experiments/maps/map_%d.txt'%map_id
            self.consider_night = False
            if tasks_id == 0:
                self.tasks = tasks.get_sequence_of_subtasks()
            if tasks_id == 1:
                self.tasks = tasks.get_interleaving_subtasks()
            if tasks_id == 2:
                self.tasks = tasks.get_safety_constraints()
                self.consider_night = True
            optimal_aux  = _get_optimal_values('../experiments/optimal_policies/map_%d.txt'%(map_id), tasks_id)

            # I store the results here
            self.results = {}
            self.optimal = {}
            self.steps = []
            for i in range(len(self.tasks)):
                self.optimal[self.tasks[i]] = learning_params.gamma ** (float(optimal_aux[i]) - 1)
                self.results[self.tasks[i]] = {}
        else:
            # Loading precomputed results
            data = read_json(file_results)
            self.results = dict([(eval(t), data['results'][t]) for t in data['results']])
            self.optimal = dict([(eval(t), data['optimal'][t]) for t in data['optimal']])
            self.steps   = data['steps']
            self.tasks   = [eval(t) for t in data['tasks']]
            # obs: json transform the interger keys from 'results' into strings
            # so I'm changing the 'steps' to strings
            for i in range(len(self.steps)):
                self.steps[i] = str(self.steps[i])

    def get_LTL_tasks(self):
        return self.tasks

    def get_task_params(self, t):
        return GameParams(self.map, t, self.consider_night)

    def run_test(self, step, sess, test_function, *test_args):
        # 'test_function' parameters should be (sess, task_params, learning_params, testing_params, *test_args)
        # and returns the reward
        for t in self.tasks:
            task_params = self.get_task_params(t)
            reward = test_function(sess, task_params, self.learning_params, self.testing_params, *test_args)
            if step not in self.results[t]:
                self.results[t][step] = []
            if len(self.steps) == 0 or self.steps[-1] < step:
                self.steps.append(step)
            self.results[t][step].append(reward)

    def show_results(self):
        average_reward = {}

        # Computing average perfomance per task
        for t in self.tasks:
            for s in self.steps:
                normalized_rewards = [r/self.optimal[t] for r in self.results[t][s]]
                a = np.array(normalized_rewards)
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]

        # Showing average perfomance across all the task
        print("\nAverage discounted reward on this map --------------------")
        print("\tsteps\tP25\tP50\tP75")
        num_tasks = float(len(self.tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            p25, p50, p75 = get_precentiles_str(normalized_rewards)
            print("\t" + str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)

    def export_results(self):
        average_reward = {}

        # Showing perfomance per task
        for t in self.tasks:
            for s in self.steps:
                normalized_rewards = [r/self.optimal[t] for r in self.results[t][s]]
                a = np.array(normalized_rewards)
                if s not in average_reward: average_reward[s] = a
                else: average_reward[s] = a + average_reward[s]

        # Computing average perfomance across all the task\
        ret = []
        num_tasks = float(len(self.tasks))
        for s in self.steps:
            normalized_rewards = average_reward[s] / num_tasks
            ret.append([s, normalized_rewards])
        return ret


class Saver:
    def __init__(self, alg_name, tester, curriculum):
        folder = "../tmp/"
        exp_name = tester.experiment
        exp_dir = os.path.join(folder, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        self.file_out = os.path.join(exp_dir, alg_name + ".json")
        self.tester = tester

    def save_results(self):
        results = {}
        results['tasks'] = [str(t) for t in self.tester.tasks]
        results['optimal'] = dict([(str(t), self.tester.optimal[t]) for t in self.tester.optimal])
        results['steps'] = self.tester.steps
        results['results'] = dict([(str(t), self.tester.results[t]) for t in self.tester.results])
        # Saving results
        save_json(self.file_out, results)


def get_precentiles_str(a):
    p25 = "%0.2f"%float(np.percentile(a, 25))
    p50 = "%0.2f"%float(np.percentile(a, 50))
    p75 = "%0.2f"%float(np.percentile(a, 75))
    return p25, p50, p75

def export_results(algorithm, task, task_id):
    for map_type, maps in [("random",range(0,5)),("adversarial",range(5,10))]:
        # Computing the summary of the results
        normalized_rewards = None
        for map_id in maps:
            result = "../tmp/task_%d/map_%d/%s.json"%(task_id, map_id, algorithm)
            tester = Tester(None, None, None, None, result)
            ret = tester.export_results()
            if normalized_rewards is None:
                normalized_rewards = ret
            else:
                for j in range(len(normalized_rewards)):
                    normalized_rewards[j][1] = np.append(normalized_rewards[j][1], ret[j][1])
        # Saving the results
        folders_out = "../results/%s/%s"%(task, map_type)
        if not os.path.exists(folders_out): os.makedirs(folders_out)
        file_out = "%s/%s.txt"%(folders_out, algorithm)
        f_out = open(file_out,"w")
        for j in range(len(normalized_rewards)):
            p25, p50, p75 = get_precentiles_str(normalized_rewards[j][1])
            f_out.write(str(normalized_rewards[j][0]) + "\t" + p25 + "\t" + p50 + "\t" + p75 + "\n")
        f_out.close()

def save_json(file, data):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

def read_json(file):
    with open(file) as data_file:
        data = json.load(data_file)
    return data

if __name__ == "__main__":

    # EXAMPLE: python3 test_utils.py --algorithm="lpopl" --tasks="sequence"

    # Getting params
    algorithms = ["dqn-l", "hrl-e", "hrl-l", "lpopl"]
    tasks      = ["sequence", "interleaving", "safety"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    parser.add_argument('--algorithm', default='lpopl', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--tasks', default='sequence', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(tasks))

    args = parser.parse_args()
    if args.algorithm not in algorithms: raise NotImplementedError("Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.tasks not in tasks: raise NotImplementedError("Tasks " + str(args.tasks) + " hasn't been defined yet")

    export_results(args.algorithm, args.tasks, tasks.index(args.tasks))

