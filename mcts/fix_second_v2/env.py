from __future__ import annotations
import numpy as np


class State:
    def __init__(
        self,
        model_constitution: list[int],
        avail_actions: dict[int, list[int]],
        model_args,
        data_args,
        training_args,
        model_info,
        models_storage,
        eval_fn,
    ) -> None:
        self.model_constitution = model_constitution
        self.avail_actions = avail_actions
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.model_info = model_info
        self.models_storage = models_storage
        self.eval_fn = eval_fn

    def __str__(self) -> str:
        str_repr = ",".join([str(block) for block in self.model_constitution])
        return str_repr

    def _get_game_end(self, new_constitution: list[int]) -> bool:
        """
        The games ends when the utility drop is greater than the threshold.
        If the game is ended, return the number of blocks that are deduped devided by number of total blocks.
        Otherwise, return 0, meaning the game isn't ended.
        """
        # Set model_path and task name for evaluation
        model_info = self.model_info
        self.model_args.model_name_or_path = model_info["model_path"]
        self.data_args.task_name = model_info["task_name"]
        acc_threshold = model_info["original_acc"] - model_info["acc_drop_threshold"]
        acc = self.eval_fn(
            self.data_args,
            self.model_args,
            self.training_args,
            model_info,
            model_constitution=new_constitution,
            model_storage=self.models_storage,
            model_id=1,
        )
        # For SVT
        if self.training_args.extra_val_eps >= 0:
            if not hasattr(self.data_args, "noise"):
                scale = 2 / (self.data_args.val_size * self.training_args.val_epsilon1)
                self.data_args.noise = np.random.laplace(loc=0, scale=scale)
                print(f"Noise to threshold: {self.data_args.noise}")
                acc_threshold += self.data_args.noise

            scale = (
                4
                * self.training_args.max_fails
                / (self.data_args.val_size * self.training_args.val_epsilon2)
            )
            noise = np.random.laplace(loc=0, scale=scale)
            print(f"Noise to acc: {noise}")
            acc += noise
        # Compare accuracy and the threshold
        if acc < acc_threshold:
            print(f"acc: {acc:.4f}, dedup success: False")
            print(f"Model constitution after dedup: {self.model_constitution}")
            return True
        else:
            print(f"acc: {acc:.4f}, dedup success: True")
            print(f"Model constitution after dedup: {new_constitution}")
            return False

    def _compute_return_value(self) -> float:
        total = len(self.model_constitution)
        n_remaining_blocks = sum(len(v) for v in self.avail_actions.values())
        return 1 - n_remaining_blocks / total

    def next_state(self, action: int, Es) -> tuple[State, float]:
        dedup_dict = dict()
        for block_2b_replace in self.avail_actions[action]:
            block_to_replace = block_2b_replace - len(self.model_constitution)
            print(f"{block_2b_replace} -> {block_to_replace}")
            dedup_dict[block_2b_replace] = block_to_replace

        # New model constitution: replace the block_2b_replaced with the action
        new_constitution = [
            dedup_dict[block] if block in dedup_dict else block
            for block in self.model_constitution
        ]

        # Check if game is ended, if games ends, return
        s = self.__str__()
        sa = f"{s}_{action}"
        if sa in Es:
            is_game_end, return_value = Es[sa]
        else:
            is_game_end = self._get_game_end(new_constitution)
            return_value = self._compute_return_value()
            Es[sa] = (is_game_end, return_value)

        if not is_game_end:
            # Update avail_actions
            del self.avail_actions[action]

        if is_game_end or len(self.avail_actions) == 0:
            return None, return_value

        next_s = State(
            new_constitution,
            self.avail_actions,
            self.model_args,
            self.data_args,
            self.training_args,
            self.model_info,
            self.models_storage,
            self.eval_fn,
        )

        return next_s, return_value
