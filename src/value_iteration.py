import math
from game_objects import *
from dfa import *

"""
It performs value iteration.
The result is directly updated in 'V'
"""
def value_iteration(S, actions, T, V, discount=1, v_init=0, e=0.01):
    # Initializing V
    for s in S:
        if s not in V: 
            V[s] = v_init
    # Learning
    while True:
        error = 0
        for s in S:
            v = V[s]
            V[s] = max([get_value_action(s,a,T,V,discount) for a in actions])
            error = max([error,abs(V[s]-v)])
        if error < e:
            break

def get_value_action(s,a,T,V,discount=1):
    return sum([T[s][a].get_probability(s2) * (T[s][a].get_reward(s2) + discount * V[s2]) for s2 in T[s][a].get_next_states()])


"""
This class saves all the information related to one transition in the MDP
"""
class Transition:
    def __init__(self, s, a):
        self.s = s     # State unique id
        self.a = a     # Action
        self.R = {}    # Sum of all the rewards received by this transition
        self.T = {}    # Dictionary where the key is the next state and the value is the counting

    # Updates the probability and reward
    def add_successor(self, s_next, prob, reward):
        self.T[s_next] = prob
        self.R[s_next] = reward

    # Returns the next states
    def get_next_states(self):
        return self.T.keys()

    # Returns the reward
    def get_reward(self, s_next):
        return float(self.R[s_next])

    # Returns the probability of transinting to s_next
    def get_probability(self, s_next):
        return float(self.T[s_next])


def evaluate_optimal_policy(map, agent_i, agent_j, consider_night, tasks, task_id):
    map_height, map_width = len(map), len(map)
    sunrise, hour_init, sunset = 5, 12, 21
    actions = [Actions.up, Actions.down, Actions.left, Actions.right]

    summary = []
    for ltl_task in tasks:
        dfa = DFA(ltl_task)
        # Creating the states
        S = set()
        # adding states without considering 'True' and 'False'
        for i in range(1,map_height-1):
            for j in range(1,map_width-1):
                if consider_night:
                    for t in range(24):
                        # I do not include states where is night and the agent is not in the shelter
                        if not(sunrise <= t <= sunset) and str(map[i][j]) != "s":
                            continue
                        for ltl in dfa.ltl2state:
                            if ltl not in ['True', 'False']:
                                S.add((i,j,t,ltl))
                else:
                    for ltl in dfa.ltl2state:
                        if ltl not in ['True', 'False']:
                            S.add((i,j,ltl))
                    
        # Adding 2 terminal states: one for 'True' and one for 'False'
        S.add('False')
        S.add('True')

        # Constructing transition and reward matrix
        T = {}
        for s in S:
            T[s] = {}
            for a in actions:
                T[s][a] = Transition(s,a)
                if s in ['False','True']:
                    T[s][a].add_successor(s, 1, 0)
                else:
                    if consider_night:
                        i,j,t,ltl = s
                    else:
                        i,j,ltl = s

                    # performing action
                    s2_i, s2_j = i, j
                    if a == Actions.up:    s2_i-=1
                    if a == Actions.down:  s2_i+=1
                    if a == Actions.left:  s2_j-=1
                    if a == Actions.right: s2_j+=1
                    if str(map[s2_i][s2_j]) == "X":
                        s2_i, s2_j = i, j
                    # Progressing time
                    if consider_night:
                        s2_t = (t+1)%24
                    # Progressing the DFA
                    true_props = str(map[s2_i][s2_j]).strip()
                    if consider_night and not(sunrise <= s2_t <= sunset):
                        true_props += "n"
                    s2_ltl = dfa.progress_LTL(ltl, true_props)
                    # Adding transition
                    if s2_ltl in ['False','True']: 
                        s2 = s2_ltl
                    else: 
                        if consider_night:
                            s2 = (s2_i,s2_j,s2_t,s2_ltl)
                        else:
                            s2 = (s2_i,s2_j,s2_ltl)

                    T[s][a].add_successor(s2, 1, -1 if s2 != 'False' else -1000)
                    if s2 not in S:
                        print("Error!")

        # Computing the optimal policy with value iteration
        V = {}
        value_iteration(S, actions, T, V)
        if consider_night:
            s = (agent_i, agent_j, hour_init, dfa.get_LTL())
        else:
            s = (agent_i, agent_j, dfa.get_LTL())
        
        summary.append(int(-V[s]))

    print(summary, "# Value optimal policy for solution experiment", task_id)

    
