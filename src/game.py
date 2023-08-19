import numpy as np
from scipy.spatial import distance
from game_objects import *
from dfa import *


class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """
    def __init__(self, map_fpath, stochastic_transition, ltl_task, consider_night, init_dfa_state, init_loc):
        self.map_fpath = map_fpath
        self.stochastic_transition = stochastic_transition
        self.ltl_task = ltl_task
        self.consider_night = consider_night
        self.init_dfa_state = init_dfa_state
        self.init_loc = init_loc


class Game:
    def __init__(self, params):
        self.params = params
        self._load_map(params.map_fpath)
        self.stochastic_transition = params.stochastic_transition
        if params.init_loc:
            self._set_agent_loc(params.init_loc)
        # Adding day and night if need it
        self.consider_night = params.consider_night
        self.hour = 12
        if self.consider_night:
            self.sunrise = 5
            self.sunset  = 21
        # Loading and progressing the LTL reward
        self.dfa = DFA(params.ltl_task, params.init_dfa_state)
        reward, self.ltl_game_over, self.env_game_over = self._get_rewards()
        self.agent.update_reward(reward)

    def execute_action(self, action):
        """
        We execute 'action' in the game
        Returns the reward that the agent gets after executing the action
        """
        agent = self.agent
        self.hour = (self.hour + 1) % 24

        # Getting new position after executing action
        if self.stochastic_transition:
            ni, nj = self._get_next_position_stochastic(action)
        else:
            ni, nj = self._get_next_position(action)

        # Interacting with the objects that is in the next position
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            # changing agent position
            agent.change_position(ni, nj)

        # Progressing the LTL reward and dealing with the consequences...
        reward, self.ltl_game_over, self.env_game_over = self._get_rewards()
        agent.update_reward(reward)

        # we continue playing
        return reward

    def _get_next_position(self, action):
        """
        Returns deterministically the position where the agent would be if we execute action
        """
        agent = self.agent
        ni, nj = agent.i, agent.j

        # OBS: Invalid actions behave as NO-OP
        if action == Actions.up   : ni-=1
        if action == Actions.down : ni+=1
        if action == Actions.left : nj-=1
        if action == Actions.right: nj+=1

        return ni, nj

    def _get_next_position_stochastic(self, action, main_prob=0.8):
        """
        Returns stochastically the position where the agent would be if we execute action.
        """
        agent = self.agent
        ni, nj = agent.i, agent.j
        side_prob = (1 - main_prob) / 3

        if action == Actions.up: probs = [main_prob, side_prob, side_prob, side_prob]
        if action == Actions.down: probs = [side_prob, main_prob, side_prob, side_prob]
        if action == Actions.left: probs = [side_prob, side_prob, main_prob, side_prob]
        if action == Actions.right: probs = [side_prob, side_prob, side_prob, main_prob]

        stochastic_action = np.random.choice([Actions.up, Actions.down, Actions.left, Actions.right], p=probs)

        # OBS: Invalid actions behave as NO-OP
        if stochastic_action == Actions.up: ni -= 1
        if stochastic_action == Actions.down: ni += 1
        if stochastic_action == Actions.left: nj -= 1
        if stochastic_action == Actions.right: nj += 1

        return ni, nj

    def _get_rewards(self):
        """
        This method progress the dfa and returns the 'reward' and if its game over
        """
        true_props = self.get_true_propositions()
        self.dfa.progress(true_props)
        reward = 1 if self.dfa.in_terminal_state() else 0
        ltl_game_over = self.dfa.is_game_over()
        env_game_over = False
        return reward, ltl_game_over, env_game_over

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = str(self.map_array[self.agent.i][self.agent.j]).strip()
        # adding the is_night proposition
        if self.consider_night and self._is_night():
            ret += "n"
        return ret

    def _is_night(self):
        return not(self.sunrise <= self.hour <= self.sunset)

    # The following methods create the map ----------------------------------------------
    def _load_map(self, map_fpath):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map
                - e.g. self.map_array[i][j]: contains the object located on row 'i' and column 'j'
            - self.agent: is the agent!
            - self.map_height: number of rows in every room
            - self.map_width: number of columns in every room
        The inputs:
            - map_fpath: path to the map file
        """
        # contains all the actions that the agent can perform
        actions = self._load_actions()
        # loading the map
        self.map_array = []
        self.class_ids = {}  # I use the lower case letters to define the features
        self.num_features = 0
        f = open(map_fpath)
        i, j = 0, 0
        for l in f:
            # I don't consider empty lines!
            if len(l.rstrip()) == 0: continue

            # this is not an empty line!
            row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    self.num_features += 1
                    entity = Empty(i, j, label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                if e in " A": entity = Empty(i, j)
                if e == "X":  entity = Obstacle(i, j)
                if e == "A":  self.agent = Agent(i, j, actions)
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array), len(self.map_array[0])

    def _load_actions(self):
        return [Actions.up, Actions.right, Actions.down, Actions.left]

    def is_valid_agent_loc(self, x, y):
        return self.map_array[x][y].interact(self.agent)

    def _set_agent_loc(self, loc):
        """
        set agent's start location instead of reading from map_x.txt
        should have checked loc is a valid agent location
        """
        self.agent.change_position(loc[0], loc[1])

    def _steps_before_dark(self):
        if self.sunrise - 1 <= self.hour <= self.sunset:
            return 1 + self.sunset - self.hour
        return 0  # it is night

    def get_num_features(self):
        """
        return the size of the feature representation of the map
        """
        if self.consider_night:
            return self.num_features + 1
        return self.num_features

    def get_features(self):
        """
        return a feature representations of the map
        """
        ret = []
        for i in range(self.map_height):
            for j in range(self.map_width):
                obj = self.map_array[i][j]
                if str(obj) in self.class_ids:  # map from object classes to numbers
                    ret.append(distance.cityblock([obj.i, obj.j], [self.agent.i, self.agent.j]))

        # Adding the number of steps before night (if need it)
        if self.consider_night:
            ret.append(self._steps_before_dark())

        return np.array(ret, dtype=np.float64)

    # def _manhattan_distance(self, obj):
    #     """
    #     Returns the Manhattan distance between 'obj' and the agent
    #     """
    #     return abs(obj.i - self.agent.i) + abs(obj.j - self.agent.j)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.agent.get_actions()

    def get_LTL_goal(self):
        """
        Returns the next LTL goal
        """
        return self.dfa.get_LTL()

    # The following methods create a string representation of the current state ---------
    def show_map(self):
        """
        Prints the current map
        """
        print(self.__str__())
        if self.consider_night:
            print("Steps before night:", self._steps_before_dark(), "Current time:", self.hour)
        print("Reward:", self.agent.reward, "Agent has", self.agent.num_keys, "keys.", "Goal", self.get_LTL_goal())

    def __str__(self):
        return self._get_map_str()

    def _get_map_str(self):
        r = ""
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if self.agent.idem_position(i, j):
                    s += str(self.agent)
                else:
                    s += str(self.map_array[i][j])
            if i > 0:
                r += "\n"
            r += s
        return r


def play(params, max_time):
    # commands
    str_to_action = {"w": Actions.up, "d": Actions.right, "s": Actions.down, "a": Actions.left}
    # play the game!
    game = Game(params)
    for t in range(max_time):
        # Showing game
        game.show_map()
        acts = game.get_actions()
        # Getting action
        print("\nSteps ", t)
        print("Action? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action and str_to_action[a] in acts:
            reward = game.execute_action(str_to_action[a])
            if game.ltl_game_over or game.env_game_over:  # Game Over
                break
        else:
            print("Forbidden action")
    game.show_map()
    return reward


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    import tasks
    map = "../experiments/maps/map_0.txt"
    # tasks = get_sequence_of_subtasks
    # tasks = tasks.get_interleaving_subtasks()
    tasks = tasks.get_safety_constraints()
    max_time = 100
    consider_night=True

    for t in tasks:
        t = tasks[-1]
        while True:
            params = GameParams(map, t, consider_night)
            if play(params, max_time) > 0:
                break
