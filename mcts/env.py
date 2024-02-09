from bisect import bisect
import pdb

from text_task_utils.evaluate import evaluate


class State:
    def __init__(
        self,
        models_constitution: list[int],
        n_unique_blocks: int,
        remaining_budgets: list[float],
        block_2b_replaced: int = -1,
        model_range: list[int] = None,
    ) -> None:
        self.models_constitution = models_constitution
        self.n_unique_blocks = n_unique_blocks
        self.remaining_budgets = remaining_budgets
        self.block_2b_replaced = block_2b_replaced
        self.model_range = model_range

        if block_2b_replaced >= 0:
            self.affected_model_ids = []
            self.affected_model_constitutions = []
            self.n_affected_blocks = 0
            self.contained_model_ids = []  # list of set of model_ids
            self._get_affected_info()

    def __str__(self) -> str:
        str_repr = ",".join([str(block) for block in self.models_constitution])
        return f"{self.block_2b_replaced}~{str_repr}"

    def get_game_end(
        self, model_storage, models_info, data_args, model_args, training_args
    ) -> int:
        """
        The games ends when the utility drop is greater than the threshold.
        If the game is ended, return negative number of blocks normalized by total number of blocks.
        Otherwise, return 0, meaning the game isn't ended.
        """
        if self.block_2b_replaced < 0:
            raise RuntimeError(
                "Shouldn't call get_game_end() when selecting a block to be replaced because it is always not the end."
            )
            # return 0

        for model_id, model_constitution in zip(
            self.affected_model_ids, self.affected_model_constitutions
        ):
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
                return reward_function(self)
        return 0

    def _block_id_to_model_id(self, block_id: int) -> int:
        return bisect(self.model_range[1:], block_id)

    def _get_affected_info(self):
        for i, start_idx in enumerate(self.model_range[:-1]):
            constitution = self.models_constitution[start_idx : self.model_range[i + 1]]
            counts = constitution.count(self.block_2b_replaced)

            if counts > 0:
                self.affected_model_ids.append(i)
                self.affected_model_constitutions.append(constitution)
                self.n_affected_blocks += counts
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
        legal_actions_1_copy: list[int],
        legal_actions_2: dict[int, dict[int, list[int]]],
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
                legal_actions.extend(legal_actions_2[self.block_2b_replaced][model_id])

            # legal_actions = chain.from_iterable(
            #     (
            #         self.models_constitution[
            #             self.model_range[model_id] : self.model_range[model_id + 1]
            #         ]
            #         for model_id in legal_model_actions
            #     )
            # )
            # legal_actions = list(set(legal_actions))
            # # Exclude the block_2b_replaced
            # legal_actions.remove(self.block_2b_replaced)
            return legal_actions
        else:
            # To choose a block to be replaced
            return legal_actions_1_copy

    def next_state(self, action, budgets: list[float] = None):
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

            # New number of unique blocks
            new_n_unique_blocks = self.n_unique_blocks - self.n_affected_blocks
            return State(
                new_constitution,
                new_n_unique_blocks,
                new_budgets,
                block_2b_replaced=-1,
                model_range=self.model_range,
            )
        else:
            print(f"Block to be replaced: {action}")
            return State(
                self.models_constitution,
                self.n_unique_blocks,
                self.remaining_budgets,
                block_2b_replaced=action,
                model_range=self.model_range,
            )


def reward_function(state: State):
    total_blocks = len(state.models_constitution)
    return 1 - (state.n_unique_blocks - state.n_affected_blocks) / total_blocks
