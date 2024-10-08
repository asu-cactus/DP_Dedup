from bisect import bisect
import pdb

from text_task_utils.evaluate import evaluate


class State:
    def __init__(
        self,
        models_constitution: list[int],
        n_unique_blocks: int,
        remaining_budgets: list[float],
        all_legal_actions: dict[int, dict[int, list[int]]],
        model_range: list[int],
        block_2b_replaced: int = -1,
        n_affected_blocks: int = 0,
        affected_model_ids: list[int] = [],
    ) -> None:
        self.models_constitution = models_constitution
        self.remaining_budgets = remaining_budgets
        self.model_range = model_range
        self.block_2b_replaced = block_2b_replaced
        # Both n_unique_blocks and n_affected_blocks are from previous 1st sub-action step.
        # n_unique_blocks value is always before accuracy test fails, so can be directly used to calculate the return value.
        self.n_unique_blocks = n_unique_blocks
        self.n_affected_blocks = n_affected_blocks
        self.affected_model_ids = affected_model_ids

        if block_2b_replaced >= 0:
            self.affected_model_ids = []
            self.n_affected_blocks = 0
            self.contained_model_ids = []  # list of set of model_ids
            self._get_affected_info()
        self.all_legal_actions = all_legal_actions

    def __str__(self) -> str:
        str_repr = ",".join([str(block) for block in self.models_constitution])
        return f"{self.block_2b_replaced}~{str_repr}"

    def get_game_end(
        self, model_storage, models_info, data_args, model_args, training_args
    ) -> int:
        """
        The games ends when the utility drop is greater than the threshold.
        If the game is ended, return the number of blocks that are deduped devided by number of total blocks.
        Otherwise, return 0, meaning the game isn't ended. In the meanwhile, update n_unique_blocks.
        """
        model_range = model_storage["model_range"]
        for model_id in self.affected_model_ids:
            model_constitution = self.models_constitution[
                model_range[model_id] : model_range[model_id + 1]
            ]
            # Set model_path and task name for evaluation
            model_args.model_name_or_path = models_info[model_id]["model_path"]
            data_args.task_name = models_info[model_id]["task_name"]
            if (
                evaluate(
                    model_storage,
                    model_id,
                    model_constitution,
                    data_args,
                    model_args,
                    training_args,
                )
                < models_info[model_id]["original_acc"]
                - models_info[model_id]["acc_drop_threshold"]
            ):

                total_blocks = self.model_range[-1]
                return 1 - self.n_unique_blocks / total_blocks
        # If the node pass the accuracy test, update n_unique_blocks and return 0
        self.n_unique_blocks -= self.n_affected_blocks
        return 0

    def _block_id_to_model_id(self, block_id: int) -> int:
        return bisect(self.model_range[1:], block_id)

    def _get_affected_info(self):
        for i, start_idx in enumerate(self.model_range[:-1]):
            constitution = self.models_constitution[start_idx : self.model_range[i + 1]]
            affected_counts = constitution.count(self.block_2b_replaced)

            if affected_counts > 0:
                self.affected_model_ids.append(i)
                self.n_affected_blocks += affected_counts
                # For each affected model, get what models it contains
                contained_models = set()
                for block_id in constitution:
                    # Each model has a model range, find model that contains the block using bisect
                    model_id = self._block_id_to_model_id(block_id)
                    contained_models.add(model_id)
                self.contained_model_ids.append(contained_models)

    def legal_actions(
        self,
        budgets: list[float],
    ) -> list[int]:

        if self.block_2b_replaced >= 0:
            # To choose a model to replace the block_2b_replaced

            # Get legal actions
            legal_model_actions = []
            for model_id in range(len(self.model_range) - 1):
                is_legal = True
                for affected_model_id in self.affected_model_ids:
                    # If candidate model has blocks that are already in the affected model, it is legal
                    contained_ids = self.contained_model_ids[
                        self.affected_model_ids.index(affected_model_id)
                    ]
                    if model_id == affected_model_id or model_id in contained_ids:
                        continue
                    # If (privacy budget of candidate model) > (remaining budget), it is illegal
                    if budgets[model_id] > self.remaining_budgets[affected_model_id]:
                        is_legal = False
                        break
                if is_legal:
                    legal_model_actions.append(model_id)
            # Get legal block actions from legal model actions
            legal_actions = []
            for model_id in legal_model_actions:
                legal_actions.extend(
                    self.all_legal_actions[self.block_2b_replaced].get(model_id, [])
                )
            # Update all_legal_actions
            # Get rid of action in all_legal_actions_copy
            # When block_2b_replaced < 0, it means we are selecting a block to be replaced
            # And the new action is block to be replaced
            del self.all_legal_actions[self.block_2b_replaced]

            return legal_actions
        else:
            # To choose a block to be replaced
            return list(self.all_legal_actions.keys())

    def next_state(
        self,
        action,
        budgets: list[float] = None,
    ):
        if self.block_2b_replaced >= 0:
            print(f"Block to replace: {action}")
            action_model = self._block_id_to_model_id(action)
            budget_loss = budgets[action_model]

            # New budget is subtracted by the budget loss if the model is affected and action_model is not in the contained models
            # pdb.set_trace()
            new_budgets = [
                (
                    budget - budget_loss
                    if kth_model in self.affected_model_ids
                    and action_model != kth_model
                    and (
                        action_model
                        not in self.contained_model_ids[
                            self.affected_model_ids.index(kth_model)
                        ]
                    )
                    else budget
                )
                for kth_model, budget in enumerate(self.remaining_budgets)
            ]

            # New model constitution: replace the block_2b_replaced with the action
            new_constitution = [
                action if block == self.block_2b_replaced else block
                for block in self.models_constitution
            ]

            # # New number of unique blocks
            # new_n_unique_blocks = self.n_unique_blocks - self.n_affected_blocks

            # Once a block is replaced, it can't replace other blocks.
            # Here, state.block_2b_replaced become the block to replace other blocks.
            to_delete_pair = []
            for block_2b_replaced, value in self.all_legal_actions.items():
                for model_id, blocks_to_replace in value.items():
                    if self.block_2b_replaced in blocks_to_replace:
                        blocks_to_replace.remove(self.block_2b_replaced)
                        if len(blocks_to_replace) == 0:
                            to_delete_pair.append((block_2b_replaced, model_id))
            to_delete = []
            for block_2b_replaced, model_id in to_delete_pair:
                del self.all_legal_actions[block_2b_replaced][model_id]
                if len(self.all_legal_actions[block_2b_replaced]) == 0:
                    to_delete.append(block_2b_replaced)
            for block_2b_replaced in to_delete:
                del self.all_legal_actions[block_2b_replaced]

            return State(
                new_constitution,
                self.n_unique_blocks,
                new_budgets,
                self.all_legal_actions,
                self.model_range,
                n_affected_blocks=self.n_affected_blocks,
                affected_model_ids=self.affected_model_ids,
            )
        else:
            print(f"Block to be replaced: {action}")
            return State(
                self.models_constitution,
                self.n_unique_blocks,
                self.remaining_budgets,
                self.all_legal_actions,
                self.model_range,
                block_2b_replaced=action,
            )
