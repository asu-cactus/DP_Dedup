import math
import pickle
import os
from random import choice
import pdb

from env import State

EPS = 1e-8


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(
        self,
        model_args,
        data_args,
        training_args,
        models_info: list[dict],
        models_storage,
        cpuct: float = math.sqrt(2),
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.models_info = models_info
        self.budgets = [info["budget"] for info in models_info]
        self.models_storage = models_storage
        self.init_state = self._get_init_state(models_storage, self.budgets)

        self.cpuct = cpuct

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Es = {}  # stores return value if is terminal, 0 otherwise
        if training_args.resume:
            self._resume()

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

    def _get_init_state(self, models_storage, budgets):
        model_range = models_storage["model_range"]
        n_unique_blocks = model_range[-1]
        models_constitution = list(range(n_unique_blocks))

        return State(
            models_constitution,
            n_unique_blocks,
            budgets,
            block_2be_replaced=-1,
            model_range=model_range,
        )

    def search(self, state: State):
        """
        This function performs one iteration/epsisode of MCTS. It is recursively
        called till a leaf node is found. The action chosen at each node is one
        that has the maximum upper confidence bound as in the paper.

        Returns:
            v: the negative of number of blocks
        """
        s = str(state)

        if state.block_2be_replaced >= 0:
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

        legal_actions = state.legal_actions(self.budgets)
        if s in self.Ns:
            # Selection: pick the action with the highest upper confidence bound
            cur_best = -float("inf")
            best_act = -1
            for a in legal_actions:
                sa = f"{s}_{a}"
                if sa in self.Qsa:
                    u = self.Qsa[sa] + self.cpuct * math.sqrt(
                        math.log(self.Ns[s]) / (self.Nsa[sa] + EPS)
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

        next_s = state.next_state(a, self.budgets)

        # Recursively search to get the return value
        v = self.search(next_s)

        # Update statistics based on the return value
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
