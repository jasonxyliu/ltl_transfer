import random, os, argparse
from collections import deque
from tasks import get_sequence_of_subtasks, get_interleaving_subtasks, get_safety_constraints
from value_iteration import evaluate_optimal_policy


def addElements(map, elements, num_per_type):
    map_height, map_width = len(map), len(map[0])
    for _ in range(num_per_type):
        for e in elements:
            while(True):
                i, j = random.randint(1, map_height-1), random.randint(1, map_width-1)
                if map[i][j] == " ":
                    map[i][j] = e
                    break
    
def getObjects(map):
    objs = {}
    for i in range(len(map)):
        for j in range(len(map[i])):
            e = map[i][j]
            if e == "A": agent = i,j
            elif e not in " X":
                if e not in objs: objs[e] = []
                objs[e].append((i,j))
    return objs, agent  

def getMD(a, o):
    return sum([abs(a[i]-o[i]) for i in range(len(a))])

def getMyopicSolution(agent, objs, task):
    if task == "": return 0
    min_cost = min([getMD(agent, pos) for pos in objs[task[0]]])
    return min([getMD(agent, pos) + getMyopicSolution(pos, objs, task[1:]) for pos in objs[task[0]] if getMD(agent, pos) == min_cost])

# This method returns a list with all the possible path's cost to solve the problem
# then you just need to take the minimum one :P
def getOptimalSolution(agent, objs, task):
    if task == "": return 0
    return min([getMD(agent, pos) + getOptimalSolution(pos, objs, task[1:]) for pos in objs[task[0]]])


def computeOptimalSolutions(map, tasks):
    # getting objects positions
    objs, agent = getObjects(map)
    # computing optimal and myopic optimal solutions
    myopic_optimal = 0
    for t in tasks:
        optimal = getOptimalSolution(agent, objs, t)
        myopic  = getMyopicSolution(agent, objs, t)
        myopic_optimal += 0.9**(myopic - optimal)
    return myopic_optimal/len(tasks)


def getAdversarialMaps(conf_params, num_adv_maps, num_eval_maps):
    min_seeds = []
    for seed in range(num_eval_maps):
        value = createMap(conf_params, seed, False)
        if len(min_seeds) < num_adv_maps:
            min_seeds.append((value, seed))
        else:
            max_seed = 0
            for i in range(1, num_adv_maps):
                if min_seeds[max_seed][0] < min_seeds[i][0]:
                    max_seed = i
            if value < min_seeds[max_seed][0]:
                min_seeds[max_seed] = (value, seed)

    for v,s in min_seeds:
        print("Seed", s, "got", v)


def createMap(conf_params, seed, show):
    # configuration parameters
    map_width, map_height, resources, fancy_resources, workstations, num_resource_per_type, num_fancy_resources_per_type, num_workstations_per_type, shelter_locations, tasks = conf_params
    random.seed(seed)
    
    # Creating a new map layout
    map = [["X"]+[" " for _ in range(map_width-2)]+["X"] for _ in range(map_height)]
    map[0] = ["X" for _ in range(map_width)]
    map[-1] = ["X" for _ in range(map_width)]

    # Adding the agent in a corner
    map[map_height//2][map_width//2] = "A"
    agent_i, agent_j = map_height//2, map_width//2

    # Adding the Shelter
    for i,j in shelter_locations:
        map[i][j] = "s"

    # Adding the work stations
    addElements(map, workstations, num_workstations_per_type)
    
    # Adding resources
    addElements(map, resources, num_resource_per_type)
    addElements(map, fancy_resources, num_fancy_resources_per_type)

    # Printing the map
    if show:
        # showing the map
        for row in map:
            print("".join(row))

        # computing optimal policies for the three set of tasks (in number of steps)
        print("Computing optimal policies using Value Iteration...")
        evaluate_optimal_policy(map, agent_i, agent_j, False, get_sequence_of_subtasks(), 1)
        evaluate_optimal_policy(map, agent_i, agent_j, False, get_interleaving_subtasks(), 2)
        evaluate_optimal_policy(map, agent_i, agent_j, True, get_safety_constraints(), 3)

    # Computing optimal and myopic optimals for each task
    return computeOptimalSolutions(map, tasks)
    

if __name__ == '__main__':

    # configuration parameters for creating a map
    map_width  = 21
    map_height = 21
    resources = 'adf'
    fancy_resources = 'gh'
    workstations = 'bce'
    num_resource_per_type = 5
    num_fancy_resources_per_type = 2
    num_workstations_per_type = 2
    shelter_locations = [(i,j) for i in range(8,13) for j in range(11,20)]
    # NOTE: the map's difficulty is measure by the difference between the reward obtained by an optimal myopic policy vs a globally optimal policy (over the sequential tasks)
    tasks = ["ab", "ac", "de", "db", "fae", "abdc", "acfb", "acfc", "faeg", "acfbh"] 
    conf_params = map_width, map_height, resources, fancy_resources, workstations, num_resource_per_type, num_fancy_resources_per_type, num_workstations_per_type, shelter_locations, tasks

    # EXAMPLE 1 (create a map): 
    #    python3 map_generator.py --create_map --seed=0

    # EXAMPLE 2 (search for adversarial seeds): 
    #    python3 map_generator.py --adversarial --num_adv_maps=5 --num_eval_maps=1000

    # Getting params
    parser = argparse.ArgumentParser(prog="map_generator", description='Generates random craft maps.')
    parser.add_argument('--create_map', help='This flag indicates that a map will be generated', action='store_true')
    parser.add_argument('--seed', default=0, type=int, help='This parameter indicates the random seed that will generate the map')
    parser.add_argument('--adversarial', help='This flag indicates that a set of random seeds will be tried in order to find adversarial maps', action='store_true')
    parser.add_argument('--num_adv_maps', default=5, type=int, help='This parameter indicates the number of adversarial maps to be displayed')
    parser.add_argument('--num_eval_maps', default=1000, type=int, help='This parameter indicates how many maps to generate when looking for adversarial maps')    
    args = parser.parse_args()

    if args.create_map:
        # Creating a new map
        print('Map computed using seed '+str(args.seed)+'...')
        createMap(conf_params, args.seed, show=True)

    if args.adversarial:
        # Collecting seeds for adversarial maps
        print('Searching for adversarial maps...')
        getAdversarialMaps(conf_params, args.num_adv_maps, args.num_eval_maps)
