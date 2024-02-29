import math
import pickle
import os
from copy import deepcopy
from random import choice
from collections import defaultdict
import pdb

from mcts.warm_start_naive.env import State
from mcts.warm_start_naive.initialize import get_heuristic_info


class MCTS:
    """
    MCTS with action pruning.
    """

    def __init__(
        self,
        model_id,
        model_args,
        data_args,
        training_args,
        models_info: list[dict],
        models_storage,
    ):
        self.model_id = model_id
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.models_info = models_info
        self.models_storage = models_storage

        # Get pruned actions
        self.all_legal_actions = get_heuristic_info(models_storage)

        self.Qsa = defaultdict(int)  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(int)  # stores #times edge s,a was visited
        self.Ns = defaultdict(int)  # stores #times board s was visited

        self.Es = {}  # stores return value if is terminal, 0 otherwise
        if training_args.resume:
            self._resume()

    def initial_episode(self, models_constitution):
        all_legal_actions = deepcopy(self.all_legal_actions)

        return State(
            self.model_id,
            models_constitution,
            all_legal_actions,
            self.model_args,
            self.data_args,
            self.training_args,
            self.models_info,
            self.models_storage,
        )

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
        legal_actions = state.legal_actions()
        if outside_tree:
            # simulation
            a = choice(legal_actions)
        elif s in self.Ns:
            # Selection: pick the action with the highest upper confidence bound
            if len(legal_actions) == 1:
                a = legal_actions[0]
            else:
                cur_best = -float("inf")
                best_act = -1
                for a in legal_actions:
                    sa = f"{s}_{a}"
                    if sa in self.Qsa:
                        u = self.Qsa[sa] + self.training_args.cprod * math.sqrt(
                            math.log(self.Ns[s]) / (self.Nsa[sa])
                        )
                        if u > cur_best:
                            cur_best = u
                            best_act = a
                    else:
                        best_act = a
                        break
                a = best_act
        else:
            # Expansion:
            a = choice(legal_actions)
            first_expanded = True
            # Next action will be outside of the current tree
            outside_tree = True

        # Get the next state
        next_s, v, block_to_replace = state.next_state(a, self.Es)
        if next_s is None:
            return v, state.model_constitution

        # Recursively search to get the return value
        v, model_constitution = self.search(next_s, outside_tree)

        # Backprogation: Update statistics based on the return value
        # Only update the nodes that are in the tree, including the newly expanded node
        if first_expanded or not outside_tree:
            sa = f"{s}_{a}_{block_to_replace}"
            self.Nsa[sa] += 1
            self.Ns[s] += 1
            self.Qsa[sa] += (v - self.Qsa[sa]) / self.Nsa[sa]
        return v, model_constitution

    def save_state(self, save_i):
        output_dir = self.training_args.output_dir
        # Combine Qsa, Nsa, Ns, Es and save to pickle file
        save_path = f"{output_dir}/Es_{save_i}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.Es, f)

        # Delete previous states if it exists
        delete_i = save_i - self.training_args.save_every * self.training_args.keep_n
        if delete_i > 0:
            delete_path = f"{output_dir}/Es_{delete_i}.pkl"
            if os.path.exists(delete_path):
                os.remove(delete_path)

    def _resume(self):
        """
        Load previous states from pickle file.
        """
        output_dir = self.training_args.output_dir
        resume_episode = self.training_args.resume_episode
        resume_path = f"{output_dir}/Es_{resume_episode}.pkl"
        with open(resume_path, "rb") as f:
            self.Es = pickle.load(f)
