# LPOPL

This project studies how to teach multiple tasks to a Reinforcement Learning (RL) agent. To this end, we use Linear Temporal Logic (LTL) as a language for specifying multiple tasks in a manner that supports the composition of learned skills. We also propose a novel algorithm that exploits LTL progression and off-policy RL to speed up learning without compromising convergence guarantees. A detailed description of our approach can be found in the following paper ([link](http://www.cs.toronto.edu/~rntoro/docs/LPOPL.pdf)):

    @inproceedings{tor-etal-aamas18,
        author = {Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A.},
        title     = {Teaching Multiple Tasks to an RL Agent using LTL},
        booktitle = {Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS)},
        year      = {2018},
        note      = {to appear}
    }

This code is meant to be a clean and usable version of our approach, called LPOPL. If you find any bugs or have questions about it, please let us know. We'll be happy to help you!


## Installation instructions

You might clone this repository by running:

    git clone https://bitbucket.org/RToroIcarte/lpopl.git

LPOPL requires [Python3.5](https://www.python.org/) with three libraries: [numpy](http://www.numpy.org/), [tensorflow](https://www.tensorflow.org/), and [sympy](http://www.sympy.org). 


## Running examples

To run LPOPL and our three baselines, move to the *src* folder and execute *run_experiments.py*. This code receives 3 parameters: The RL algorithm to use (which might be "dqn-l", "hrl-e", "hrl-l", or "lpopl"), the tasks to solve (which might be "sequence", "interleaving", "safety"), and the map (which is an integer between -1 and 9). Maps 0 to 4 were randomly generated. Maps 5 to 9 are adversarial maps. Select '--map=-1' to run experiments over the 10 maps with three trials per map. For instance, the following command solves the 10 *sequence tasks* over map 0 using LPOPL:

    python3 run_experiments.py --algorithm="lpopl" --tasks="sequence" --map=0

The results will be printed and saved in './tmp'. After running LPOPL over all the maps, you might run *test_util.py* (which also receives the algorithm and task parameters) to compute the average performance across the 10 maps:

    python3 test_util.py --algorithm="lpopl" --tasks="sequence"

The overall results will be saved in the './results' folder.


## Acknowledgments

Our implementation of LPOPL is based on the DQN baseline code provided by [OpenAI](https://github.com/openai/baselines). We encourage you to check out their repository. They are doing really cool RL stuff too :)
