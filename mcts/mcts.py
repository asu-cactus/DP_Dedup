import math
import pickle
import os
from copy import deepcopy
from random import choice
import pdb

from mcts.env import State
from mcts.heuristics import get_heuristic_info

EPS = 1e-8


class MCTS:
    """
    MCTS with action pruning.
    """

    def __init__(
        self,
        model_args,
        data_args,
        training_args,
        models_info: list[dict],
        models_storage,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.models_info = models_info
        self.budgets = [info["budget"] for info in models_info]
        self.models_storage = models_storage

        # Get pruned actions
        (
            _,
            _,
            _,
            self.all_legal_actions,
            # self.legal_actions_reverse,
        ) = get_heuristic_info(
            model_args,
            data_args,
            training_args,
            models_info,
            models_storage,
        )

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Es = {}  # stores return value if is terminal, 0 otherwise
        if training_args.resume:
            self._resume()

    def initial_episode(self):
        model_range = self.models_storage["model_range"]
        n_unique_blocks = model_range[-1]
        models_constitution = list(range(n_unique_blocks))
        # An action will be removed from legal_actions_1_copy after a block is replaced
        # self.legal_actions_1_copy = self.legal_actions_1.copy()

        return State(
            models_constitution,
            n_unique_blocks,
            self.budgets,
            deepcopy(self.all_legal_actions),
            block_2b_replaced=-1,
            model_range=model_range,
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

        # # Check if the current state is the end of the game
        # if len(state.all_legal_actions) == 0:
        #     # If there is no block to be replaced, then the game ends
        #     return reward_function(state)

        # If the state.block_2b_replaced < 0, then we know it is not the end of the game
        if state.block_2b_replaced >= 0:
            if s not in self.Es:
                self.Es[s] = state.get_game_end(
                    self.models_storage,
                    self.models_info,
                    self.data_args,
                    self.model_args,
                    self.training_args,
                )

            if self.Es[s] != 0:
                # terminal node
                return self.Es[s]

        # Selection, Expansion, Simulation, Backpropagation
        first_expanded = False
        legal_actions = state.legal_actions(self.budgets)
        if outside_tree:
            # simulation
            a = choice(legal_actions)
        elif s in self.Ns:
            # Selection: pick the action with the highest upper confidence bound
            cur_best = -float("inf")
            best_act = -1
            for a in legal_actions:
                sa = f"{s}_{a}"
                if sa in self.Qsa:
                    u = self.Qsa[sa] + math.sqrt(
                        2 * math.log(self.Ns[s]) / (self.Nsa[sa] + EPS)
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
        next_s = state.next_state(a, self.budgets)

        # Recursively search to get the return value
        v = self.search(next_s, outside_tree)

        # Backprogation: Update statistics based on the return value
        if first_expanded or not outside_tree:
            sa = f"{s}_{a}"
            # The following is according to alpha-zero-general implementation:
            # https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
            # self.Qsa[sa] = (self.Nsa.get(sa, 0) * self.Qsa.get(sa, 0) + v) / (
            #     self.Nsa.get(sa, 0) + 1
            # )

            # According to the classicial MCTS, reference:
            # https://www.cs.utexas.edu/~pstone/Courses/394Rspring11/resources/mcrave.pdf.
            self.Nsa[sa] = self.Nsa.get(sa, 0) + 1
            self.Ns[s] = self.Ns.get(s, 0) + 1
            if sa in self.Qsa:
                self.Qsa[sa] += (v - self.Qsa[sa]) / self.Nsa[sa]
            else:
                self.Qsa[sa] = v / self.Nsa[sa]
        return v

    def save_state(self, save_i, delete_i):
        output_dir = self.training_args.output_dir
        # Combine Qsa, Nsa, Ns, Es and save to pickle file
        save_dict = {
            "Qsa": self.Qsa,
            "Nsa": self.Nsa,
            "Ns": self.Ns,
            "Es": self.Es,
        }
        save_path = f"{output_dir}/mcts_states_{save_i}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

        # Delete previous states if it exists
        if delete_i > 0:
            delete_path = f"{output_dir}/mcts_states_{delete_i}.pkl"
            if os.path.exists(delete_path):
                os.remove(delete_path)

    def _resume(self):
        """
        Load previous states from pickle file.
        """
        output_dir = self.training_args.output_dir
        resume_episode = self.training_args.resume_episode
        resume_path = f"{output_dir}/mcts_states_{resume_episode}.pkl"
        with open(resume_path, "rb") as f:
            resume_dict = pickle.load(f)
        self.Qsa = resume_dict["Qsa"]
        self.Nsa = resume_dict["Nsa"]
        self.Ns = resume_dict["Ns"]
        self.Es = resume_dict["Es"]
