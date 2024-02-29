import re
from dataclasses import dataclass
import pdb


@dataclass
class Step:
    block_to_replace: int
    block_2b_replaced: int
    acc: float
    success: bool = False


def visualize_dedupicable_sequence(baseline_filename):
    deup_sequence = []
    with open(baseline_filename, "r") as f:
        for i, line in enumerate(f):
            if i > 154 and i % 6 == 5:
                m = re.search("block (.+?) -> (.+?) acc: (.+)", line)
                block_to_replace = int(m.group(1))
                block_2b_replaced = int(m.group(2))
                acc = float(m.group(3))
                deup_sequence.append(
                    Step(block_to_replace, block_2b_replaced, acc, acc >= 0.87)
                )
    # Save the dedup sequence
    with open(f"{baseline_filename[:-4]}_dedup_seq.out", "w") as f:
        for step in deup_sequence:
            f.write(
                f"{step.block_2b_replaced} -> {step.block_to_replace} acc: {step.acc} success: {step.success}\n"
            )


def parse_final_constitution():
    from final_constitution_quantile import model0, model1

    # from final_constitution_l2norm import model0, model1

    print(f"model 0 len: {len(model0)}")
    print(f"model 1 len: {len(model1)}")
    for ith_model, (model, start_idx) in enumerate([(model0, 0), (model1, 833)]):
        dedup_count = 0
        for i, block in enumerate(model):
            if block == i + start_idx:
                model[i] = 0
            else:

                dedup_count += 1
                if block >= start_idx:
                    model[i] = 1
                else:
                    model[i] = 2
        print(f"model {ith_model}: {model}")
        print(f"dedup_count: {dedup_count}")


if __name__ == "__main__":
    # parse_final_constitution()

    visualize_dedupicable_sequence("baseline3_75quantile.out")
    visualize_dedupicable_sequence("baseline3_l2norm.out")
