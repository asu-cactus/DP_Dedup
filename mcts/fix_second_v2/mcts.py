import math
from copy import deepcopy
from random import choice, random
from collections import defaultdict
import pdb

from mcts.fix_second_v2.env import State


class MCTS:
    def __init__(
        self,
        model_args,
        data_args,
        training_args,
        model_info,
        models_storage,
        eval_fn,
        train_fn,
        action_space,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.model_info = model_info
        self.models_storage = models_storage
        self.eval_fn = eval_fn
        self.train_fn = train_fn
        self.action_space = action_space

        self.Qsa = defaultdict(int)  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(int)  # stores #times edge s,a was visited
        self.Ns = defaultdict(int)  # stores #times board s was visited

        self.Es = {}  # stores return value if is terminal, 0 otherwise

    def initial_episode(self):
        model_start = self.models_storage["model_range"][1]
        model_end = self.models_storage["model_range"][2]
        model_constitution = list(range(model_start, model_end))

        avail_actions = deepcopy(self.action_space)
        return State(
            model_constitution,
            avail_actions,
            self.model_args,
            self.data_args,
            self.training_args,
            self.model_info,
            self.models_storage,
            self.eval_fn,
            self.train_fn,
        )

    # def _default_policy(self, legal_actions: list[Action]):
    #     """
    #     This function is called when the search reaches a leaf node. It returns
    #     the value of the state from the perspective of the current player.
    #     """
    #     # Get action that has the lowest norm
    #     best_action = min(legal_actions, key=lambda x: x.block_2b_replaced_norm)
    #     # With 80% probability, choose the best action, otherwise choose randomly
    #     if random() < 0.8:
    #         return best_action
    #     else:
    #         return choice(legal_actions)

    def search(self, state: State, outside_tree: bool = False):
        """
        This function performs one iteration/epsisode of MCTS. It is recursively
        called till a leaf node is found. The action chosen at each node is one
        that has the maximum upper confidence bound as in the paper.

        Returns:
            v: the negative of number of blocks
        """
        s = str(state)

        # Selection, Expansion, Simulation, Backpropagation
        first_expanded = False
        legal_actions = list(state.avail_actions)
        if outside_tree:
            # simulation
            a = choice(legal_actions)
        elif s in self.Ns:
            # Selection: pick the action with the highest upper confidence bound
            best_u = -float("inf")
            best_a = -1
            for a in legal_actions:
                sa = f"{s}_{a}"
                if sa in self.Qsa:
                    u = self.Qsa[sa] + self.training_args.cprod * math.sqrt(
                        math.log(self.Ns[s]) / (self.Nsa[sa])
                    )
                    if u > best_u:
                        best_u = u
                        best_a = a
                else:
                    best_a = a
                    break
            a = best_a
        else:
            # Expansion:
            a = choice(legal_actions)
            first_expanded = True
            # Next action will be outside of the current tree
            outside_tree = True

        # Get the next state
        next_s, v = state.next_state(a, self.Es)

        if next_s is not None:
            v = self.search(next_s, outside_tree)

        # Backprogation: Update statistics based on the return value
        # Only update the nodes that are in the tree, including the newly expanded node
        if first_expanded or not outside_tree:
            sa = f"{s}_{a}"
            self.Nsa[sa] += 1
            self.Ns[s] += 1
            self.Qsa[sa] += (v - self.Qsa[sa]) / self.Nsa[sa]
        return v
