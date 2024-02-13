import math
import pickle
from random import choice
import os
from copy import deepcopy
import pdb

from mcts.env import State
from mcts.heuristics import get_heuristic_info

EPS = 1e-8


class MCTS:
    """
    MCTS with Heuristic MC-RAVE.
    """

    def __init__(
        self,
        model_args,
        data_args,
        training_args,
        models_info: list[dict],
        models_storage,
    ):
        # Put inputs as attributes
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.models_info = models_info
        self.budgets = [info["budget"] for info in models_info]
        self.models_storage = models_storage

        # Initialize the some constants
        # Heuristic function (a constant)
        (
            self.H,
            self.H1a_to_C1a,
            self.H2aa_to_C2aa,
            self.all_legal_actions,
            # self.legal_actions_reverse,
        ) = get_heuristic_info(
            model_args,
            data_args,
            training_args,
            models_info,
            models_storage,
        )

        # Equivalence parameter
        self.k = training_args.equivalence_param

        # Initialize the MCTS tree
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Q1a = {}  # Store heuristic value for each 1st-stage action
        self.N1a = {}  # Store heuristic confidence for each 1st-stage action
        self.Q2aa = {}  # Store heuristic value for each 2nd-stage action
        self.N2aa = {}  # Store heuristic confidence for each 2nd-stage action

        self.Es = {}  # stores return value if is terminal, 0 otherwise
        self.As = {}  # stores the set of actions that have been taken from state s
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
            model_range=model_range,
        )

    def _apply_heuristics(self, state: State, legal_actions: list[int]):
        """Apply heuristics after expansion.
        Initialize the (Qsa, Nsa, Ns) and ((Q1a, N1a) or (Q2aa,N2aa)) using the heuristic function.

        """
        s = str(state)
        block_2b_replaced = state.block_2b_replaced
        for a in legal_actions:
            sa = f"{s}_{a}"
            self.Qsa[sa] = self.H
            self.Nsa[sa] = 1

            if block_2b_replaced < 0:
                self.Q1a[str(a)] = self.H
                self.N1a[str(a)] = self.H1a_to_C1a[a]
                # self.Nsa[sa] = self.H1a_to_C1a[a]
            else:
                aa = f"{block_2b_replaced}~{a}"
                self.Q2aa[aa] = self.H
                self.N2aa[aa] = self.H2aa_to_C2aa[block_2b_replaced][a]
                # self.Nsa[sa] = self.H2aa_to_C2aa[block_2b_replaced][a]
        # Update self.Ns
        self.Ns[s] = sum(self.Nsa[f"{s}_{a}"] for a in legal_actions)

    def search(self, state: State, outside_tree: bool = False):
        """
        This function performs one iteration/epsisode of MCTS. It is recursively
        called till a leaf node is found. The action chosen at each node is one
        that has the maximum upper confidence bound as in the paper.

        Returns:
            v: the negative of number of blocks
        """
        s = str(state)

        # Check if the current state is the end of the game
        if state.block_2b_replaced < 0:
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
            # simulation using the default policy
            a = choice(legal_actions)
        elif s in self.Ns:
            # Selection with Hand-selection schedule in MC RAVE
            if len(legal_actions) == 1:
                a = legal_actions[0]
            else:
                beta = math.sqrt(self.k / (3 * self.Ns[s] + self.k))
                cur_best = -float("inf")
                best_act = -1
                for a in legal_actions:
                    sa = f"{s}_{a}"
                    if state.block_2b_replaced < 0:
                        Q_heuristic = self.Q1a[str(a)]
                    else:
                        aa = f"{state.block_2b_replaced}~{a}"
                        Q_heuristic = self.Q2aa[aa]
                    u = (1 - beta) * self.Qsa[sa] + beta * Q_heuristic
                    u += self.training_args.cprod * math.sqrt(
                        math.log(self.Ns[s]) / (self.Nsa[sa] + EPS)
                    )
                    if u > cur_best:
                        cur_best = u
                        best_act = a
                a = best_act
        else:
            # Expansion:
            self._apply_heuristics(state, legal_actions)
            a = choice(legal_actions)
            first_expanded = True
            # Next action will be outside of the current tree
            outside_tree = True

        # Get the next state
        next_s = state.next_state(a, self.budgets)

        # Recursively search to get the return value
        v = self.search(next_s, outside_tree)

        # Backprogation: Update statistics based on the return value
        # Only update the nodes that are in the tree
        if first_expanded or not outside_tree:
            sa = f"{s}_{a}"
            self.Nsa[sa] += 1
            self.Ns[s] += 1
            self.Qsa[sa] += (v - self.Qsa[sa]) / self.Nsa[sa]

            # Update the Q1a and N1a or Q2aa and N2aa, depending on the value of block_2b_replaced
            if state.block_2b_replaced < 0:
                # Update the Q1a and N1a when current state selects a block to be replace
                a = str(a)
                self.N1a[a] += 1
                self.Q1a[a] += (v - self.Q1a[a]) / self.N1a[a]
            else:
                # Update the Q2aa and N2aa when current state selects a block to be replaced
                aa = f"{state.block_2b_replaced}~{a}"
                self.N2aa[aa] += 1
                self.Q2aa[aa] += (v - self.Q2aa[aa]) / self.N2aa[aa]
        return v

    def save_state(self, save_i, delete_i):
        output_dir = self.training_args.output_dir
        # Combine Qsa, Nsa, Ns, Es and save to pickle file
        save_dict = {
            "Qsa": self.Qsa,
            "Nsa": self.Nsa,
            "Ns": self.Ns,
            "Es": self.Es,
            "Q1a": self.Q1a,
            "N1a": self.N1a,
            "Q2aa": self.Q2aa,
            "N2aa": self.N2aa,
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
        self.Q1a = resume_dict["Q1a"]
        self.N1a = resume_dict["N1a"]
        self.Q2aa = resume_dict["Q2aa"]
        self.N2aa = resume_dict["N2aa"]
        self.Es = resume_dict["Es"]
