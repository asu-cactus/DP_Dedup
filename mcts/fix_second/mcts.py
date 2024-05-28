import math
import pickle
import os
from copy import deepcopy
from random import choice, random
from collections import defaultdict
import pdb

from mcts.dynamic_pruning_v2.env import State, Action
from mcts.dynamic_pruning_v2.pruning import get_heuristic_info


class MCTS:
    """
    MCTS with action pruning.
    """

    def __init__(
        self,
        model_args,
        data_args,
        training_args,
        model_info,
        models_storage,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.model_info = model_info
        self.models_storage = models_storage

        self.Qsa = defaultdict(int)  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(int)  # stores #times edge s,a was visited
        self.Ns = defaultdict(int)  # stores #times board s was visited

        self.Es = {}  # stores return value if is terminal, 0 otherwise
        if training_args.resume:
            self._resume()

    def initial_episode(self):
        total_blocks = self.models_storage["model_range"][-1]
        models_constitution = list(range(total_blocks))
        all_legal_actions = deepcopy(self.all_legal_actions)

        return State(
            models_constitution,
            all_legal_actions,
            self.model_args,
            self.data_args,
            self.training_args,
            self.model_info,
            self.models_storage,
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

    def search(
        self,
        state: State,
        outside_tree: bool = False,
        steps_before_eval: int = None,
    ):
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
        legal_actions = state.legal_actions(fanout)
        if outside_tree:
            # simulation
            a = choice(legal_actions)
        elif s in self.Ns:
            # Selection: pick the action with the highest upper confidence bound
            best_u = -float("inf")
            best_a = -1
            for a in legal_actions:
                sa = f"{s}_{a.block_2b_replaced}"
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
        if steps_before_eval < 0:
            steps_before_eval = self.training_args.eval_every - 1
        next_s, curr_v = state.next_state(a, steps_before_eval, self.Es)

        if next_s is not None:
            # If game is not ended, recursively search to get the return value; gradually increase fanout
            steps_before_eval -= 1
            fanout += 1
            v, steps_to_fail = self.search(
                next_s, fanout, outside_tree, steps_before_eval
            )
        else:
            # If game is ended, return the return value
            # curr_v should not be used, because it is a failed case.
            steps_to_fail = 0
            return curr_v, steps_to_fail

        steps_to_fail += 1
        # Backprogation: Update statistics based on the return value
        # Only update the nodes that are in the tree, including the newly expanded node
        if steps_to_fail >= self.training_args.eval_every:
            # print("Update nodes")
            if first_expanded or not outside_tree:
                sa = f"{s}_{a.block_2b_replaced}"
                self.Nsa[sa] += 1
                self.Ns[s] += 1
                self.Qsa[sa] += (v - self.Qsa[sa]) / self.Nsa[sa]

        if steps_to_fail == self.training_args.eval_every:
            # Return the value of the last passed test
            # print(f"Return value: {curr_v}")
            return curr_v, steps_to_fail
        else:
            return v, steps_to_fail
