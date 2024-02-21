from __future__ import annotations
from bisect import bisect
from dataclasses import dataclass
import pdb

from text_task_utils.evaluate import evaluate


@dataclass
class Action:
    block_2b_replaced: int
    block_to_replace: int
    models_affected: list[int]


class State:
    def __init__(
        self,
        models_constitution: list[int],
        all_legal_actions: dict[int, dict[int, list[int]]],
        model_args,
        data_args,
        training_args,
        models_info: list[dict],
        models_storage,
    ) -> None:
        self.models_constitution = models_constitution
        self.all_legal_actions = all_legal_actions
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.models_info = models_info
        self.models_storage = models_storage

        self.model_range = self.models_storage["model_range"]

        # Get contained_model_ids and remaining_budgets, used for repeated legal action check.
        allowed_budgets = [info["allowed_budget"] for info in self.models_info]
        self.training_budgets = [info["budget"] for info in self.models_info]

        self.contained_model_ids = []
        self.contained_block_ids = []
        self.remaining_budgets = []
        for model_id, model_range_start in enumerate(self.model_range[:-1]):
            model_range_end = self.model_range[model_id + 1]
            self.contained_block_ids.append(
                set(models_constitution[model_range_start:model_range_end])
            )
            # Get contained model ids
            contained_model_ids = set()
            for block in models_constitution[model_range_start:model_range_end]:
                origin_model_id = self._block_id_to_model_id(block)
                contained_model_ids.add(origin_model_id)
            self.contained_model_ids.append(contained_model_ids)

            # Get remaining budgets
            allowed_budget = allowed_budgets[model_id]
            for contained_model_id in contained_model_ids:
                allowed_budget -= self.training_budgets[contained_model_id]
            self.remaining_budgets.append(allowed_budget)

    def _block_id_to_model_id(self, block_id: int) -> int:
        return bisect(self.model_range[1:], block_id)

    def __str__(self) -> str:
        str_repr = ",".join([str(block) for block in self.models_constitution])
        return str_repr

    def _legal_model_actions(self, block_2b_replaced):
        # Get models that are affected by the block_2b_replaced
        models_affected = []
        for model_to_place, contained_block_ids in enumerate(self.contained_block_ids):
            if block_2b_replaced in contained_block_ids:
                models_affected.append(model_to_place)

        # Get legal models to replace the block_2b_replaced
        legal_model_actions = []
        for model_to_replace, budget_loss in enumerate(self.training_budgets):
            is_legal = True
            for model_affected in models_affected:
                # If candidate model (model_to_replace) has blocks that are already in the affected model, it is legal
                if model_to_replace in self.contained_model_ids[model_affected]:
                    continue
                # If (privacy budget of candidate model) > (remaining budget), it is illegal
                if budget_loss > self.remaining_budgets[model_affected]:
                    is_legal = False
                    break
            if is_legal:
                legal_model_actions.append(model_to_replace)
        return legal_model_actions, models_affected

    def legal_actions(
        self,
        fanout: int,
    ) -> list[Action]:
        legal_actions = []
        for block_2b_replaced, action_infos in self.all_legal_actions.items():
            legal_model_actions, models_affected = self._legal_model_actions(
                block_2b_replaced
            )

            for action_info in action_infos:
                if (
                    action_info.block_to_replace in self.all_legal_actions
                    and action_info.model_id in legal_model_actions
                ):
                    legal_actions.append(
                        Action(
                            block_2b_replaced=block_2b_replaced,
                            block_to_replace=action_info.block_to_replace,
                            models_affected=models_affected,
                        )
                    )
                    break

        # Get top k (fanout) legal actions
        legal_actions = legal_actions[:fanout]
        # return only the block to be replaced
        return legal_actions

    def _get_game_end(
        self,
        action: Action,
        new_constitution: list[int],
    ) -> float:
        """
        The games ends when the utility drop is greater than the threshold.
        If the game is ended, return the number of blocks that are deduped devided by number of total blocks.
        Otherwise, return 0, meaning the game isn't ended.
        """
        n_unique_blocks = len(set(new_constitution))
        return_value = 1 - n_unique_blocks / self.model_range[-1]
        # for model_id, model_range_start in enumerate(self.model_range[:-1]):
        for model_id in action.models_affected:
            model_range_start = self.model_range[model_id]
            model_range_end = self.model_range[model_id + 1]
            model_constitution = new_constitution[model_range_start:model_range_end]

            # Set model_path and task name for evaluation
            model_info = self.models_info[model_id]
            self.model_args.model_name_or_path = model_info["model_path"]
            self.data_args.task_name = model_info["task_name"]
            if (
                evaluate(
                    self.models_storage,
                    model_id,
                    model_constitution,
                    self.data_args,
                    self.model_args,
                    self.training_args,
                )
                < model_info["original_acc"] - model_info["acc_drop_threshold"]
            ):

                return True, return_value
        return False, return_value

    def next_state(
        self,
        action: Action,
        steps_before_eval: int,
        Es,
    ) -> tuple[State, float]:
        print(f"{action.block_2b_replaced} -> {action.block_to_replace}")
        # New model constitution: replace the block_2b_replaced with the action
        new_constitution = [
            action.block_to_replace if block == action.block_2b_replaced else block
            for block in self.models_constitution
        ]

        # Check if game is ended, if games ends, return
        return_value = -1
        if steps_before_eval == 0:
            s = self.__str__()
            sa = f"{s}_{action.block_2b_replaced}"
            if sa in Es:
                is_game_end, return_value = Es[sa]
            else:
                is_game_end, return_value = self._get_game_end(action, new_constitution)
                Es[sa] = (is_game_end, return_value)
            # print(f"return_value: {return_value}")
            if is_game_end:
                return None, return_value

        # Block that is replaced can't be replaced again
        del self.all_legal_actions[action.block_2b_replaced]

        next_s = State(
            new_constitution,
            self.all_legal_actions,
            self.model_args,
            self.data_args,
            self.training_args,
            self.models_info,
            self.models_storage,
        )

        return next_s, return_value
