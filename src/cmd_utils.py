import argparse
from dataclasses import dataclass


def str2bool(v: str) -> bool:
    """
    Accepts a range of true/false spellings so users can pass
    1/0, true/false, yes/no, on/off, etc.
    """
    if isinstance(v, bool):  # already a bool
        return v
    truthy = {"yes", "true", "t", "1", "y", "on"}
    falsy = {"no", "false", "f", "0", "n", "off"}
    v_lower = v.lower()
    if v_lower in truthy:
        return True
    if v_lower in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


@dataclass
class Config:
    use_random_shuffle: bool = True
    print_intermediate_info: bool = True
    entropy_threshold: float = 1.5
    batch_num_for_full_ldpc: list = None
    record_intermediate_batches: bool = False
    max_additional_batches: int = 28
    try_fix_unreliable_on_last_batch: bool = False
    use_non_zero_inequality: bool = True
    keys_to_test: int = 1
    base_oracle_calls: int = 7
    seed: int = 100
    simulate_oracle: bool = True
    port: int = 3334
    enable_full_rotation: bool = True
    print_diff: bool = True


def get_config(argv) -> Config:
    p = argparse.ArgumentParser(
        description="Runner of LDPC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--use_random_shuffle",
        type=str2bool,
        default=Config.use_random_shuffle,
        help="Use random checks instead of creating a random graph without cycles of length 4",
    )
    p.add_argument(
        "--print_intermediate_info",
        type=str2bool,
        default=Config.print_intermediate_info,
        help="Print detailed progress",
    )
    p.add_argument(
        "--record_intermediate_batches",
        type=str2bool,
        default=Config.record_intermediate_batches,
        help="Record the output as if we stopped earlier on every batch",
    )
    p.add_argument(
        "--try_fix_unreliable_on_last_batch",
        type=str2bool,
        default=Config.try_fix_unreliable_on_last_batch,
        help="Adding checks constructed from unreliable variables on last batch. This calls full LDPC after last batch",
    )
    p.add_argument(
        "--use_non_zero_inequality",
        type=str2bool,
        default=Config.use_non_zero_inequality,
        help="If true, try to search for inequalities with non-zero threshold",
    )
    p.add_argument(
        "--simulate_oracle",
        type=str2bool,
        default=Config.simulate_oracle,
        help="Instead of using the actual oracle, simulate it",
    )
    p.add_argument(
        "--enable_full_rotation",
        type=str2bool,
        default=Config.enable_full_rotation,
        help="Use precomputed encodings over full rotation checks",
    )
    p.add_argument(
        "--print_diff",
        type=str2bool,
        default=Config.print_diff,
        help="Print differences from the secret key",
    )

    p.add_argument(
        "--entropy_threshold",
        type=float,
        default=Config.entropy_threshold,
        help="Ignore checks with total entropy less than this value",
    )
    p.add_argument(
        "--batch_num_for_full_ldpc",
        type=int,
        nargs="+",
        help="Batch index (indices) at which we run full LDPC",
    )
    p.add_argument(
        "--max_additional_batches",
        type=int,
        default=Config.max_additional_batches,
        help="Number of batches to add",
    )
    p.add_argument(
        "--keys_to_test",
        type=int,
        default=Config.keys_to_test,
        help="Number of keys to recover",
    )
    p.add_argument(
        "--base_oracle_calls",
        type=int,
        default=Config.base_oracle_calls,
        help="Number of oracle calls during full rotation checks",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=Config.seed,
        help="Random seed for reproduction",
    )
    p.add_argument(
        "--port",
        type=int,
        default=Config.port,
        help="If oracle is not simulated, start listener on defined port to communicate (i.e. send ciphertext, get a 0/1 result)",
    )

    args = p.parse_args(argv)  # let argparse pull from sys.argv when argv is None
    return Config(**vars(args))
