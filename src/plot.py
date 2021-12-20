import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def make_plot(task):
    algo_colors = {
        "dqn-l": ["b", "royalblue"],
        "hrl-e": ["y", "khaki"],
        "hrl-l": ["deepskyblue", "skyblue"],
        "lpopl": ["r", "lightcoral"]
    }

    results = {}
    results_dpath = os.path.join("results", task)
    # create 2 subplots for random and adversarial maps
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[12, 5])
    xs = range(1000, 101000, 1000)
    for map_type, ax in zip(["random", "adversarial"], [ax1, ax2]):
        results[map_type] = {}
        results_fpath = os.path.join(results_dpath, map_type)
        for algo in algorithms:
            if algo != "lpopl":
                continue
            results[map_type][algo] = defaultdict(list)
            results_fpath = os.path.join(results_fpath, algo+".txt")
            with open(results_fpath, "r") as rf:
                lines = rf.readlines()
                for line in lines:
                    line_items = line.strip().split("\t")
                    results[map_type][algo]["25"].append(float(line_items[1]))
                    results[map_type][algo]["50"].append(float(line_items[2]))
                    results[map_type][algo]["75"].append(float(line_items[3]))

            ax.fill_between(xs, results[map_type][algo]["75"], results[map_type][algo]["25"],
                            facecolor=algo_colors[algo][1], alpha=0.25)
            ax.plot(xs, results[map_type][algo]["50"], color=algo_colors[algo][0])
            ax.set_xlim(left=0, right=100000)
            ax.ticklabel_format(style="sci", scilimits=(5, 5))
            ax.set_xticks(np.arange(20000, 120000, 20000))
            ax.set_ylim(bottom=0)
            ax.yaxis.tick_right()
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_title("5 %s maps"%map_type)
            ax.set_xlabel("Number of training steps")
            ax.set_ylabel("Normalized reward")
    for ax in ax1, ax2:
        ax.grid(True)
    # plt.show()
    plt.savefig(os.path.join("results", task))


if __name__ == "__main__":
    algorithms = ["dqn-l", "hrl-e", "hrl-l", "lpopl"]
    tasks = ["sequence", "interleaving", "safety"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    parser.add_argument('--tasks', default='sequence', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(tasks))

    args = parser.parse_args()
    if args.tasks not in tasks: raise NotImplementedError("Tasks " + str(args.tasks) + " hasn't been defined yet")

    make_plot(args.tasks)
