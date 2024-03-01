from __future__ import annotations
import pdb

from text_task_utils.evaluate import evaluate

DEFAULT_RETURN_V = 0


class State:
    def __init__(
        self,
        model_id: int,
        model_constitution: list[int],
        all_legal_actions: dict[int, dict[int, list[int]]],
        model_args,
        data_args,
        training_args,
        model_info: dict,
        models_storage,
    ) -> None:
        self.model_id = model_id
        self.model_constitution = model_constitution
        self.all_legal_actions = all_legal_actions
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.model_info = model_info
        self.models_storage = models_storage

        model_range = self.models_storage["model_range"]
        self.model_range_start = model_range[model_id]
        self.model_range_end = model_range[model_id + 1]
        self.model_nblocks = self.model_range_end - self.model_range_start

    def __str__(self) -> str:
        str_repr = ",".join([str(block) for block in self.model_constitution])
        return str_repr

    def legal_actions(self) -> list[int]:
        return list(self.all_legal_actions.keys())

    def _get_game_end(
        self,
        model_constitution: list[int],
    ) -> float:
        """
        The games ends when the utility drop is greater than the threshold.
        If the game is ended, return the number of blocks that are deduped devided by number of total blocks.
        Otherwise, return 0, meaning the game isn't ended.
        """
        # Set model_path and task name for evaluation

        self.model_args.model_name_or_path = self.model_info["model_path"]
        self.data_args.task_name = self.model_info["task_name"]
        if (
            evaluate(
                self.models_storage,
                self.model_id,
                model_constitution,
                self.data_args,
                self.model_args,
                self.training_args,
            )
            < self.model_info["original_acc"] - self.model_info["acc_drop_threshold"]
        ):
            n_unique_blocks_in_cur_model = 0
            for block_id in set(model_constitution):
                if (
                    block_id >= self.model_range_start
                    and block_id < self.model_range_end
                ):
                    n_unique_blocks_in_cur_model += 1
            # Add 1 more block because this step failed to dedup
            n_unique_blocks_in_cur_model += 1
            v = 1 - n_unique_blocks_in_cur_model / self.model_nblocks
            return v
        return DEFAULT_RETURN_V

    def next_state(
        self,
        action,
        Es,
        n_candidates: int = 3,
    ) -> tuple[State, float]:
        s = self.__str__()
        sa = f"{s}_{action}"
        n = 0
        for block_to_replace in self.all_legal_actions[action]:
            print(f"{action} -> {block_to_replace}")
            # New model constitution: replace the block_2b_replaced with the action
            new_constitution = [
                block_to_replace if block == action else block
                for block in self.model_constitution
            ]
            v = Es[sa] if sa in Es else self._get_game_end(new_constitution)
            # print(f"return_value: {v}")
            if v == 0:
                break

            n += 1
            if n == n_candidates:
                break

        Es[sa] = v
        # Block that is replaced can't be replaced again
        del self.all_legal_actions[action]

        next_s = (
            State(
                self.model_id,
                new_constitution,
                self.all_legal_actions,
                self.model_args,
                self.data_args,
                self.training_args,
                self.model_info,
                self.models_storage,
            )
            if v == DEFAULT_RETURN_V
            else None
        )
        # when v == DEFAULT_RETURN_V, the game isn't ended, next_s is None
        # when v != DEFAULT_RETURN_V, the game is ended, next_s is None
        return next_s, v
