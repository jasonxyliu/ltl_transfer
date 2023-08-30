# LTL-Transfer

This work shows ways to reuse policies trained to solve a set of training tasks, specified by linear temporal logic (LTL), to solve novel LTL tasks in a zero-shot manner.
Please see the following paper for more details.

Skill Transfer for Temporally-Extended Task Specifications [[Liu, Shah, Rosen, Konidaris, Tellex 2022]](https://arxiv.org/abs/2206.05096)


## Installation instructions

You might clone this repository by running:

    git clone https://github.com/jasonxyliu/ltl_transfer.git

Training state-centric policies with LPOPL requires [Python3.5](https://www.python.org/) with three libraries: [numpy](http://www.numpy.org/), [tensorflow](https://www.tensorflow.org/), and [sympy](http://www.sympy.org).
Python 3.7 should also work.

Transfer Learning requires [dill](https://dill.readthedocs.io/en/latest/), [NetworkX](https://networkx.org/), [Matplotlib](https://matplotlib.org/), and [mpi4py](https://mpi4py.readthedocs.io/en/stable/) if use on a cluster.

Visualization requires [pillow](https://pillow.readthedocs.io/en/stable/index.html)

Install all dependencies in a conda environment by running the following command

    conda create -n ltl_transfer python=3.7 numpy sympy dill networkx matplotlib pillow tensorflow=1  # tensorflow 1.15


## Running examples
Navigation into *src* folder then run *run_experiments.py*.

To run LPOPL to learn state-centric policies

    python3 run_experiments.py --algo=lpopl --train_type=mixed --train_size=50 --map=0 --prob=0.7 --total_steps=800000

To run zero-shot transfer on a local machine

    python run_experiments.py --algo=zero_shot_transfer --train_type=mixed --train_size=50 --test_type=soft --map=0 --prob=0.7 --relabel_method=local

Reduce ```RELABEL_CHUNK_SIZE``` to 21 in ``transfer.py`` if run the above Python script slows down your machine too much. It controls how many parallel processes are running at a time.

To run zero-shot transfer on a cluster

    python run_experiments.py --algo=zero_shot_transfer --train_type=mixed --train_size=50 --test_type=soft --map=0 --relabel_method=cluster


## Visualization

To visualize initiation set classifiers

    python visualize_classifiers.py --algo=lpopl --tasks_id=4 --map_id=0 --ltl_id=12 --simple_vis
    

## Generating new random maps

You might generate new random maps using the code in *src/map_generator.py*. The only parameter required is the random seed to be used. The resulting map will be displayed in the console along with the number of steps that an optimal policy would need to solve the "sequence", "interleaving", and "safety" tasks (this value is computed using value iteration and might take a few minutes):

    python3 map_generator.py --create_map --seed=0

It is also possible to automatically look for adversarial maps for the Hierarchical RL baseline. To do so, we generate *num_eval_maps* random maps and rank them according to the difference between the reward obtained by an optimal policy and the reward obtained by an optimal myopic policy. The code will display the random seeds of the top *num_adv_maps* ranked maps. (You might then display those maps using the *--create_map* flag.)

    python3 map_generator.py --adversarial --num_adv_maps=5 --num_eval_maps=1000

## Acknowledgments
Our implementation is developed on top of the LPOPL [codebase](https://bitbucket.org/RToroIcarte/lpopl/src/master/) 

Please let us know if you spot any bug or have any question. We are happy to help!
