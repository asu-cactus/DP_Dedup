from __future__ import annotations
import pdb


class State:
    def __init__(
        self,
        model_constitution: list[int],
        avail_actions: list[int],
        model_args,
        data_args,
        training_args,
        model_info,
        models_storage,
        eval_fn,
        train_fn,
    ) -> None:
        self.model_constitution = model_constitution
        self.avail_actions = avail_actions
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.model_info = model_info
        self.models_storage = models_storage
        self.eval_fn = eval_fn
        self.train_fn = train_fn

    def __str__(self) -> str:
        str_repr = ",".join([str(block) for block in self.model_constitution])
        return str_repr

    def _get_game_end(
        self,
        # action: int,
        new_constitution: list[int],
    ) -> bool:
        """
        The games ends when the utility drop is greater than the threshold.
        If the game is ended, return the number of blocks that are deduped devided by number of total blocks.
        Otherwise, return 0, meaning the game isn't ended.
        """
        # Set model_path and task name for evaluation
        model_info = self.model_info
        self.model_args.model_name_or_path = model_info["model_path"]
        self.data_args.task_name = model_info["task_name"]
        if (
            self.eval_fn(
                self.data_args,
                self.model_args,
                self.training_args,
                model_info,
                model_constitution=new_constitution,
                model_storage=self.models_storage,
                model_id=1,
            )
            < model_info["original_acc"] - model_info["acc_drop_threshold"]
        ):
            return True
        else:
            return False

    def next_state(
        self,
        action: int,
        steps_before_eval: int,
        Es,
    ) -> tuple[State, float]:
        block_to_replace = action - len(self.model_constitution)
        print(f"{action} -> {block_to_replace}")
        # New model constitution: replace the block_2b_replaced with the action
        new_constitution = [
            block_to_replace if block == action else block
            for block in self.model_constitution
        ]

        # Check if game is ended, if games ends, return
        return_value = -1
        if steps_before_eval == 0:

            s = self.__str__()
            sa = f"{s}_{action}"
            if sa in Es:
                is_game_end, return_value = Es[sa]
            else:
                is_game_end = self._get_game_end(new_constitution)
                return_value = 1 - len(self.avail_actions) / len(new_constitution)
                # if is_game_end:
                #     return_value = 1 - len(self.avail_actions) / len(new_constitution)
                # else:
                #     return_value = 1 - (len(self.avail_actions) - 1) / len(
                #         new_constitution
                #     )
                Es[sa] = (is_game_end, return_value)

            if is_game_end:
                return None, return_value

        # Update avail_actions
        avail_actions = [a for a in self.avail_actions if a != action]

        next_s = State(
            new_constitution,
            avail_actions,
            self.model_args,
            self.data_args,
            self.training_args,
            self.model_info,
            self.models_storage,
            self.eval_fn,
            self.train_fn,
        )

        return next_s, return_value
