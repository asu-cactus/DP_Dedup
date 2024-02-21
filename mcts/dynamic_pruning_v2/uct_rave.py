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
        models_info: list[dict],
        models_storage,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.models_info = models_info
        self.models_storage = models_storage

        # Get pruned actions
        self.all_legal_actions = get_heuristic_info(
            model_args,
            data_args,
            training_args,
            models_info,
            models_storage,
        )

        # Initialize the MCTS tree
        self.Qsa = defaultdict(int)  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(int)  # stores #times edge s,a was visited
        self.Ns = defaultdict(int)  # stores #times board s was visited

        # Store heuristic value for each 1st-stage action
        self.Q1sa = defaultdict(int)
        self.N1sa = defaultdict(int)

        # Equivalence parameter
        self.k = training_args.equivalence_param

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
            self.models_info,
            self.models_storage,
        )

    def search(
        self,
        state: State,
        fanout: int,
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
            beta = math.sqrt(self.k / (3 * self.Ns[s] + self.k))
            best_u = -float("inf")
            best_a = -1
            for a in legal_actions:
                sa = f"{s}_{a.block_2b_replaced}"
                Q_rave = self.Q1sa[sa]
                if sa in self.Qsa:
                    u = (1 - beta) * self.Qsa[sa] + beta * Q_rave
                    u += self.training_args.cprod * math.sqrt(
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
            sa = f"{s}_{a.block_2b_replaced}"
            if first_expanded or not outside_tree:
                self.Nsa[sa] += 1
                self.Ns[s] += 1
                self.Qsa[sa] += (v - self.Qsa[sa]) / self.Nsa[sa]

            # Update the Q1sa and N1sa
            # Update for every action even the node is outside of the tree
            self.N1sa[sa] += 1
            self.Q1sa[sa] += (v - self.Q1sa[sa]) / self.N1sa[sa]

        if steps_to_fail == self.training_args.eval_every:
            # Return the value of the last passed test
            # print(f"Return value: {curr_v}")
            return curr_v, steps_to_fail
        else:
            return v, steps_to_fail

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
