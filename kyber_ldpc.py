import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ.setdefault(v, "1")

import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ.setdefault(v, "1")

import argparse
import itertools as it
import pickle
import random
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from math import comb, log, prod

import numpy as np
import psutil

sys.path.append("../SCA-LDPC/simulate-with-python")

import platform

from simulate.kyber import sample_secret_coefs, secret_distribution
from simulate.make_code import generate_ldpc_from_protograph
from simulate.max_likelihood import (
    SimpleOracle,
    pr_cond_yx,
    s_distribution_for_all_y,
    s_distribution_from_hard_y,
)

from src.adaptive_search_for_encoding import (
    MILP_create_model,
    MILP_run_model,
    classify_pmf,
    efficient_joint_pmfs,
    most_promising_direction,
    most_promising_two_directions,
    precompute_mask_and_directions,
    primitive_tuple,
)
from src.kyber_encodings import (
    ETA,
    KYBER_K,
    SMALLEST_THRESHOLD,
    Z_BOUND,
    anticyclic_shift_multi,
    build_z_values_arr,
    center,
    compress,
    decompress,
    du,
    encoding_for_compound_split,
    entropy,
    get_inequalities,
    inequalities_to_split,
    k,
    marginal_pmf,
    marginal_pmfs,
    multibit_encoding,
    n,
    potential_z_coefs,
    secret_joint_range,
    split_from_canonical,
)
from src.kyber_oracle import (
    KyberOracle,
    build_arbitrary_combination_ciphertext,
    build_full_rotate_ciphertext,
    read_sk,
)
from src.ldpc import (
    bit_tuple_to_int,
    compute_pairs,
    generate_ldpc_block,
    generate_ldpc_matrix,
    ldpc_decode,
    sample_coef_static,
)

if platform.system() == "Darwin" and (
    platform.machine() == "arm64" or platform.processor() == "arm"
):
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    QOS_CLASS_USER_INTERACTIVE = 0x21  # see <pthread/qos.h>
    if libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0:
        raise OSError(ctypes.get_errno(), "pthread_set_qos_class_self_np failed")


def timed_query(oracle, ct):
    t_wall_0 = time.perf_counter()
    t_cpu_0 = time.process_time()  # user+sys time for *this* process
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    y = oracle.query(ct)
    t_wall_1 = time.perf_counter()
    t_cpu_1 = time.process_time()
    r1 = resource.getrusage(resource.RUSAGE_SELF)

    wall_ms = (t_wall_1 - t_wall_0) * 1e3
    cpu_ms = (t_cpu_1 - t_cpu_0) * 1e3  # user+sys
    vol_cs = r1.ru_nvcsw - r0.ru_nvcsw  # voluntary context switches
    print(f"wall={wall_ms:7.3f} ms  cpu={cpu_ms:7.3f} ms  voluntary_ctx_sw={vol_cs}")
    return y


def is_in(x, lst: list):
    return lst is not None and x in lst


def list_small_str(lst, precision=3):
    lst_str = ", ".join(f"{val:.{precision}f}" for val in lst)
    return "[" + lst_str + "]"


def has_length_4_cycle(res_idxs):
    # Get the number of check variables
    num_check_vars = len(res_idxs)

    # Iterate over all pairs of check variables (i, j)
    for i in range(num_check_vars):
        for j in range(i + 1, num_check_vars):
            # Find the intersection of the secret variables for check i and check j
            common_vars = set(res_idxs[i]).intersection(res_idxs[j])

            # Check if there are 2 or more common secret variables
            if len(common_vars) >= 2:
                print(i, j, res_idxs[i], res_idxs[j])
                return True  # A length-4 cycle is found

    return False  # No length-4 cycles found


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


cfg = get_config(sys.argv[1:])

p = 0.95
pr_oracle = SimpleOracle(p)

sk_len = n * KYBER_K
assert k == KYBER_K or k == KYBER_K, "Need to add support for Kyber1024"

joint_weight = 4
prob_s = secret_distribution(ETA)
joint_pmf = list()
for s in secret_joint_range(ETA, weight=joint_weight):
    joint_pmf.append(prod(prob_s[si] for si in s))

s_prior = list(list(prob_s.values()) for _ in range(sk_len))
secret_variables = np.array(s_prior, dtype=np.float32)

if cfg.enable_full_rotation:
    database_dir = f"database_10000_{sk_len}"

    full_rotation_check = (0, 64, 128, 192)
    full_rotation_checks = []
    for block_idx in range(k):
        for i in range(64):
            full_rotation_checks.append(
                tuple(x + i + n * block_idx for x in full_rotation_check)
            )
    database_4_full = eval(
        open(os.path.join(database_dir, "database_4_full.txt"), "rt").read()
    )

    if KYBER_K == 2:
        if cfg.base_oracle_calls == 6:
            conf = ([520, 1573, 1364, 471, 97, 196], [6, 4, 2, 6, 3, 1])
        elif cfg.base_oracle_calls == 7:
            conf = ([2185, 520, 1573, 1364, 471, 97, 196], [0, 6, 4, 2, 6, 3, 1])
        elif cfg.base_oracle_calls == 8:
            conf = (
                [449, 2185, 520, 1573, 1364, 471, 97, 196],
                [1, 0, 6, 4, 2, 6, 3, 1],
            )
        elif cfg.base_oracle_calls == 9:
            conf = (
                [1704, 449, 2185, 520, 1573, 1364, 471, 97, 196],
                [7, 1, 0, 6, 4, 2, 6, 3, 1],
            )
        else:
            raise ValueError("Unsupported number of calls for base case")
    if KYBER_K == 3:
        if cfg.base_oracle_calls == 6:
            conf = ([149, 543, 45, 312, 30, 4], [4, 2, 1, 3, 4, 6])
        elif cfg.base_oracle_calls == 7:
            conf = ([25, 149, 543, 45, 312, 30, 4], [3, 4, 2, 1, 3, 4, 6])
        elif cfg.base_oracle_calls == 8:
            conf = ([465, 25, 149, 543, 45, 312, 30, 4], [0, 3, 4, 2, 1, 3, 4, 6])
        elif cfg.base_oracle_calls == 9:
            conf = (
                [201, 465, 25, 149, 543, 45, 312, 30, 4],
                [5, 0, 3, 4, 2, 1, 3, 4, 6],
            )
        elif cfg.base_oracle_calls == 10:
            conf = (
                [523, 201, 465, 25, 149, 543, 45, 312, 30, 4],
                [1, 5, 0, 3, 4, 2, 1, 3, 4, 6],
            )
        else:
            raise ValueError("Unsupported number of calls for base case")
    full_rotation_inequalities = get_inequalities(
        conf,
        database_4_full,
    )
    full_rotation_split = inequalities_to_split(full_rotation_inequalities)

max_info_per_call = 1 - entropy([p, 1 - p])
theoretical_num_of_calls = n * k * entropy(prob_s) / max_info_per_call
print(f"Theory suggest that the minimum number of calls is {theoretical_num_of_calls}")

# mask, Z = precompute_mask_and_directions(
#     first_z=5, s_bound=ETA, joint_weight=joint_weight
# )
first_z = 5
S = np.array(list(secret_joint_range(bound=ETA, weight=joint_weight)), dtype=int)
small_z_subset = set(potential_z_coefs[:first_z]) | set(
    -x for x in potential_z_coefs[:first_z]
)
small_z_subset = list(small_z_subset)
Z = np.array(
    list(set(z for z in it.product(small_z_subset, repeat=joint_weight) if any(z)))
)
dot = S @ Z.T
mask = dot > 0
mask = mask.T

if cfg.use_non_zero_inequality:
    # Just trying to do something reasonable? But seems random still
    coef_mult = 12
    if KYBER_K == 2:
        z_coefs_list = [-14, -12, -10, -8, -5, -3, 0, 3, 5, 8, 10, 12, 14]
    elif KYBER_K == 3:
        z_coefs_list = [-10, -8, -5, -3, 0, 3, 5, 8, 10]
    Z_t = np.array(
        list(
            set(
                tuple(
                    np.vectorize(decompress)(
                        np.vectorize(compress)(np.array(z) * coef_mult, du), du
                    )
                )
                for z in it.product(
                    z_coefs_list,
                    repeat=joint_weight,
                )
                if any(z)
            )
        )
    )
    Z_t = center(Z_t)
    Z_t = Z_t[np.sum(abs(Z_t), axis=1) < Z_BOUND]
    dot_t = S @ Z_t.T
    mask_t = dot_t > SMALLEST_THRESHOLD
    mask_t = mask_t.T

coef_support_size = len(prob_s)
differences_arr = []
differences_for_batches = []
oracle_calls_for_batches = []
total_oracle_calls = 0
# lost_info_arr = []

configuration_probs = defaultdict(list)
time_cond_pr = 0
time_ldpc = 0
time_ldpc_batch = 0
time_batches_total = 0
time_base = 0
time_base_marginals = 0
time_direction_enumeration = 0

test_keys = cfg.keys_to_test
additional_batches = cfg.max_additional_batches
batch_checks_num = sk_len // joint_weight

random.seed(10)
if not cfg.use_random_shuffle:
    additional_checks = generate_ldpc_matrix(
        sk_len, joint_weight, additional_batches, full_rotation_checks, random
    )
# print(additional_checks)

# print(has_length_4_cycle(full_rotation_checks + additional_checks))
# exit()
# additional_checks = []

# proto_path = "proto.txt"
# with open(proto_path, "wt") as f:
#     print(f"{additional_batches} {joint_weight}", file=f)
#     print("dense", file=f)
#     for _ in range(additional_batches):
#         print(" ".join(map(str, [1] * joint_weight)), file=f)
# # precomputed seed that result in success first try
# if additional_batches == 1:
#     initial_seed = 24
# elif additional_batches == 2:
#     initial_seed = 24
# elif additional_batches == 3:
#     initial_seed = 61
# elif additional_batches == 4:
#     initial_seed = 223
# elif additional_batches == 5:
#     initial_seed = 245
# else:
#     initial_seed = 0
# while True:
#     random.seed(initial_seed)
#     additional_checks = generate_ldpc_from_protograph(proto_path, batch_checks_num)
#     # if not check_length_4_cycle_for_additional_checks(additional_checks):
#     if has_length_4_cycle(additional_checks):
#         initial_seed += 1
#     else:
#         break
# print(f"succeded with seed {initial_seed}")

# if has_length_4_cycle(full_rotation_checks + additional_checks):
#     print("a cycle!")
# else:
#     print("no cycles")
# exit()

# for _ in range(additional_batches):
#     new_block, used_pairs = generate_ldpc_block(
#         set([(174, 490), (170, 593)]),
#         n_rows=batch_checks_num,
#         row_weight=joint_weight,
#         rng=random,
#     )
#     additional_checks.extend(new_block)
#     seen_pairs = used_pairs

# print("Precomputation is done, going to the recovery")
random.seed(cfg.seed)

if not cfg.simulate_oracle and cfg.keys_to_test > 1:
    raise ValueError("Currently can recover only one key not from simulation")

if not cfg.simulate_oracle:
    oracle = KyberOracle("127.0.0.1", cfg.port)

ct_info = open("ct_info.txt", "wt")
y_statistic = defaultdict(int)
for key_idx in range(test_keys):
    intermediate_sk = []
    differences_for_batch = []
    oracle_calls_for_batch = []
    unreliable_idxs = []
    # each 7 calls of oracle loses about 0.31941 bits compared to theory
    # lost_information = 0.31941 * (sk_len // joint_weight)

    # Either create a new secret key to simulate calls or obtain a real one for
    # difference computation
    if cfg.simulate_oracle:
        sk = sample_secret_coefs(sk_len)
    else:
        # we should only go here only after long search of compound evsets are done,
        # so, secret key should be already written to file
        sk = read_sk("../GoFetch/poc/crypto_victim/kyber.txt")
    all_checks = []
    check_variables = []
    pr_oracle.oracle_calls = 0

    sk_decoded_marginals = [0] * sk_len

    if cfg.enable_full_rotation:
        t0 = time.perf_counter_ns()
        checks, split = full_rotation_checks, full_rotation_split
        check_encoding = multibit_encoding(split)
        pickled_filename = f"cond_prob_all_k{k}_y_0{int(p * 100)}_{len(split)}bits"
        if p == 0.95 and len(split) in [6, 7, 8, 9, 10]:
            if os.path.exists(pickled_filename):
                with open(pickled_filename, "rb") as f:
                    all_y_pmf, pr_y = pickle.load(f)
            else:
                all_y_pmf, pr_y = s_distribution_for_all_y(
                    pr_oracle, check_encoding, joint_pmf
                )
                with open(pickled_filename, "wb") as f:
                    pickle.dump((all_y_pmf, pr_y), f)
        else:
            all_y_pmf, pr_y = s_distribution_for_all_y(
                pr_oracle, check_encoding, joint_pmf
            )

        all_checks.extend(checks)
        for check_idxs in checks:
            if cfg.simulate_oracle:
                enc_idx = 0
                for var_idx in check_idxs:
                    enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
                x = check_encoding[enc_idx]
                y = sample_coef_static(x, pr_oracle)
            else:
                y = []
                for inequality in full_rotation_inequalities:
                    z_values, thresholds, enabled, signs = inequality
                    ct = build_full_rotate_ciphertext(
                        z_values,
                        joint_weight,
                        thresholds,
                        enabled,
                        signs,
                        SMALLEST_THRESHOLD,
                        check_idxs[0] // n,
                        check_idxs[0] % n,
                        oracle,
                    )
                    response = oracle.query(ct)
                    y.append(response)
            y_idx = bit_tuple_to_int(y)
            pmf = all_y_pmf[y_idx]

            channel_pmf = np.array(
                list(pr_cond_yx(y, x, pr_oracle) for x in check_encoding)
            )
            channel_pmf /= sum(channel_pmf)
            check_variables.append(channel_pmf)

            # check_variables.append(pmf)
            t1 = time.perf_counter_ns()
            s_marginal = marginal_pmf(pmf, joint_weight)
            for i, var_idx in enumerate(check_idxs):
                sk_decoded_marginals[var_idx] = s_marginal[i]
            time_base_marginals += time.perf_counter_ns() - t1
        time_base += time.perf_counter_ns() - t0

        if cfg.print_intermediate_info:
            base_bad_variables = []
            for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded_marginals)):
                if actual_pmf[expect + ETA] < 0.1:
                    base_bad_variables.append(i)
            print(f"After base case got bad variables: {base_bad_variables}")
    else:
        sk_decoded_marginals = secret_variables

    t2 = time.perf_counter_ns()
    for batch_no in range(additional_batches):
        if cfg.print_intermediate_info:
            print(f"{batch_no=}")
        if cfg.use_random_shuffle:
            checks = list(range(sk_len))
            random.shuffle(checks)
            checks = list(
                sorted(checks[i * joint_weight : (i + 1) * joint_weight])
                for i in range(batch_checks_num)
            )
        else:
            checks = additional_checks[
                batch_no * batch_checks_num : (batch_no + 1) * batch_checks_num
            ]

        for check_pos_in_batch, check_idxs in enumerate(checks):
            check_idxs = sorted(check_idxs)
            pmfs = list(sk_decoded_marginals[i] for i in check_idxs)
            pmfs_entropy = list(map(entropy, pmfs))
            if sum(pmfs_entropy) < cfg.entropy_threshold:
                continue
            # record the checks for LDPC
            all_checks.append(check_idxs)
            joint_pmf_conditional = efficient_joint_pmfs(pmfs)
            # pr_y, z_values_arr, encoding = most_promising_two_directions(
            #     mask, Z, joint_pmf_conditional
            # )

            # enc_idx = 0
            # for var_idx in check_idxs:
            #     enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
            # x = encoding[enc_idx]
            # y = sample_coef_static(x, pr_oracle)
            # y_statistic[y] += 1

            # channel_pmf = np.array(list(pr_cond_yx(y, x, pr_oracle) for x in encoding))
            # channel_pmf /= sum(channel_pmf)
            # check_variables.append(channel_pmf)

            # posterior_pmf = s_distribution_from_hard_y(
            #     y, pr_oracle, encoding, joint_pmf
            # )

            t0 = time.perf_counter_ns()
            pr, z_values, encoding = most_promising_direction(
                mask, Z, joint_pmf_conditional
            )
            threshold = 0
            # arbitrary value, but try to improve the result if we are too far from 0.5
            if cfg.use_non_zero_inequality and abs(pr - 0.5) > 0.05:
                # TODO: can only consider indices where probs > 0.5 since we can only decrease
                # pr of 1; thus, can do less multiplications, potential speed up
                probs_t = mask_t @ joint_pmf_conditional
                best_pr_idx = abs(probs_t - 0.5).argmin()
                pr_t = probs_t[best_pr_idx]
                if abs(pr_t - 0.5) < abs(pr - 0.5):
                    z_new = Z_t[best_pr_idx]
                    if cfg.print_intermediate_info:
                        print(
                            f"Managed to improve pr from {pr:.4f} to {pr_t:.4f} looking at non-zero inequality; old z: {z_values}, new z: {z_new}"
                        )
                    pr = pr_t
                    z_values = z_new
                    encoding = mask_t[best_pr_idx]
                    threshold = 1
                elif cfg.print_intermediate_info:
                    print(
                        f"{pr:.4f} stays unimproved with non-zero inequality, best new = {pr_t}"
                    )

            time_direction_enumeration += time.perf_counter_ns() - t0
            # lost_information += abs(pr - 0.5)

            if cfg.simulate_oracle:
                enc_idx = 0
                for var_idx in check_idxs:
                    enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
                x = encoding[enc_idx]
                y = pr_oracle.predict_bit(x, 0)
            else:
                ct = build_arbitrary_combination_ciphertext(
                    z_values,
                    joint_weight,
                    threshold,
                    SMALLEST_THRESHOLD,
                    check_idxs,
                    oracle,
                )
                y = oracle.query(ct)
                if batch_no == 0:
                    enc_idx = 0
                    for var_idx in check_idxs:
                        enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
                    x = encoding[enc_idx]
                    print(
                        f"{check_pos_in_batch}: secret variables: {check_idxs}",
                        file=ct_info,
                    )
                    print(
                        f"x==y?:{x == y}; {x=}, {y=}",
                        file=ct_info,
                    )
                    print(
                        f"dot: {np.dot(z_values, list(sk[var_idx] for var_idx in check_idxs))}",
                        file=ct_info,
                    )
                    print(", ".join(f"0x{b:02x}" for b in ct), file=ct_info)

            y_statistic[y] += 1
            channel_pmf = np.array(list(pr_oracle.prob_of(x, y, 0) for x in encoding))
            channel_pmf /= sum(channel_pmf)
            check_variables.append(channel_pmf)

            if cfg.print_intermediate_info:
                labels = list(classify_pmf(pmf) for pmf in pmfs)
                decreasing_pr = sorted(joint_pmf_conditional, reverse=True)
                # if batch_no == 1:
                ### regex:: taking indices \[.*170.*\];
                print(
                    f"taking indices {check_idxs}; {labels=}; largest probs={list_small_str(decreasing_pr[:4], 3)}"
                )
                print(f"pmfs=[{','.join(map(lambda x: list_small_str(x, 5), pmfs))}]")
                print(
                    f"entropy={list_small_str(pmfs_entropy, 4)}, total entropy={sum(pmfs_entropy):.4f}"
                )
                print(f"sk coefs={list(sk[var_idx] for var_idx in check_idxs)}")
                print(f"x==y?:{x == y}; {x=}, {y=}")
                # s_marginal = marginal_pmf(posterior_pmf, joint_weight)
                # print(
                #     f"improved marginal=[{','.join(map(lambda x: list_small_str(x, 5), s_marginal))}]"
                # )
                print(f"Simple enumeration of directions obtain {pr=} for {z_values=}")
                # print(
                #     f"Simple enumeration of 2 directions obtain {pr_y} for {z_values_arr=}, entropy={entropy(pr_y)}"
                # )
                print("============")

            # configuration_probs[tuple(labels)].append(pr)

        # if batch_no == additional_batches - 1:
        #     break
        t0 = time.perf_counter_ns()
        check_variables_intermediate = np.array(check_variables, dtype=np.float32)
        epsilon = 1e-20
        check_variables_intermediate[check_variables_intermediate == 0] = epsilon
        if batch_no == additional_batches - 1 and cfg.print_intermediate_info:
            sk_decoded_marginals = ldpc_decode(
                all_checks,
                secret_variables,
                check_variables_intermediate,
                joint_weight,
                10000,
                ETA,
                layered=False,
            )
        # here we try to use good LDPC call in the middle
        elif is_in(batch_no, cfg.batch_num_for_full_ldpc):
            sk_decoded_marginals = ldpc_decode(
                all_checks,
                secret_variables,
                check_variables_intermediate,
                joint_weight,
                10000,
                ETA,
                layered=False,
            )
        else:
            # if batch_no == additional_batches - 1:
            #     iterations = 2
            # else:
            if cfg.use_random_shuffle:
                iterations = 2
            else:
                iterations = 3
            sk_decoded_marginals = ldpc_decode(
                all_checks,
                secret_variables,
                check_variables_intermediate,
                joint_weight,
                iterations,
                ETA,
                layered=True,
            )

        t1 = time.perf_counter_ns()
        time_ldpc_batch += t1 - t0
        if cfg.print_intermediate_info:
            intermediate_sk.append(sk_decoded_marginals)
            print(f"Spend on batch LDPC: {float((t1 - t0) / 1000000)}")

        if cfg.record_intermediate_batches:
            sk_decoded = ldpc_decode(
                all_checks,
                secret_variables,
                check_variables_intermediate,
                joint_weight,
                10000,
                ETA,
                layered=False,
            )
            oracle_calls_for_batch.append(pr_oracle.oracle_calls)
            batch_diff = 0
            for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded)):
                actual = np.argmax(actual_pmf) - ETA
                if expect != actual:
                    batch_diff += 1
            differences_for_batch.append(batch_diff)

        if batch_no == additional_batches - 1:
            if cfg.print_intermediate_info:
                print("Unreliable entries on last batch:")
                for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded_marginals)):
                    if max(actual_pmf) < 0.8:
                        print(f"{i}: s_i = {expect}  {list_small_str(actual_pmf)}")
        # print(f"LDPC intermediate output after batch {batch_no}:")
        # for i, (expect, pmf) in enumerate(zip(sk, sk_decoded_marginals)):
        #     print(f"{i}: sk_i={expect}, {list_small_str(pmf)}")

    if cfg.try_fix_unreliable_on_last_batch:
        if cfg.print_intermediate_info:
            print("Incorrect variables before adding additional last batch:")
            for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded_marginals)):
                actual = np.argmax(actual_pmf) - ETA
                if expect != actual:
                    print(f"{i}: {expect=}, {actual=}, {list_small_str(actual_pmf)}")
        sk_decoded_marginals = ldpc_decode(
            all_checks,
            secret_variables,
            check_variables_intermediate,
            joint_weight,
            10000,
            ETA,
            layered=False,
        )
        for i, actual_pmf in enumerate(sk_decoded_marginals):
            if max(actual_pmf) < 0.8:
                unreliable_idxs.append(i)
        if len(unreliable_idxs) % joint_weight != 0:
            unreability = 0.81
            to_add = joint_weight - (len(unreliable_idxs) % joint_weight)
            while to_add != 0:
                for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded_marginals)):
                    if max(actual_pmf) < unreability and i not in unreliable_idxs:
                        unreliable_idxs.append(i)
                        to_add -= 1
                        if to_add == 0:
                            break
                unreability += 0.02
        if cfg.print_intermediate_info:
            print(f"Adding last batch checks: {unreliable_idxs}")
        random.shuffle(unreliable_idxs)
        checks = list(
            sorted(unreliable_idxs[i * joint_weight : (i + 1) * joint_weight])
            for i in range(0, len(unreliable_idxs) // joint_weight)
        )
        for check_idxs in checks:
            pmfs = list(sk_decoded_marginals[i] for i in check_idxs)
            all_checks.append(check_idxs)
            joint_pmf_conditional = efficient_joint_pmfs(pmfs)
            pr_y, z_values_arr, encoding = most_promising_two_directions(
                mask, Z, joint_pmf_conditional
            )

            if cfg.simulate_oracle:
                enc_idx = 0
                for var_idx in check_idxs:
                    enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
                x = encoding[enc_idx]
                y = sample_coef_static(x, pr_oracle)
            else:
                y = []
                for z_values in z_values_arr:
                    ct = build_arbitrary_combination_ciphertext(
                        z_values,
                        joint_weight,
                        0,
                        SMALLEST_THRESHOLD,
                        check_idxs,
                        oracle,
                    )
                    response = oracle.query(ct)
                    y.append(response)
            for y_val in y:
                y_statistic[y_val] += 1

            channel_pmf = np.array(list(pr_cond_yx(y, x, pr_oracle) for x in encoding))
            channel_pmf /= sum(channel_pmf)
            check_variables.append(channel_pmf)
    time_batches_total += time.perf_counter_ns() - t2

    # with open("sk_marginals", "wb") as f:
    #     pickle.dump(sk_decoded_marginals, f)

    # additional_check = all_checks[193]
    # print(f"{additional_check=}")
    # sk_subset = list(sk[var_idx] for var_idx in additional_check)
    # print(f"{sk_subset=}")
    # s_marginal = marginal_pmf(check_variables[193], joint_weight)
    # print(f"marginal=[{','.join(map(lambda x: list_small_str(x, 5), s_marginal))}]")

    # print("Trying to ignore base oracle calls")
    # all_checks = all_checks[64 * 3 :]
    # check_variables = check_variables[64 * 3 :]

    t0 = time.perf_counter_ns()
    check_variables = np.array(check_variables, dtype=np.float32)
    epsilon = 1e-20
    check_variables[check_variables == 0] = epsilon
    sk_decoded = ldpc_decode(
        all_checks,
        secret_variables,
        check_variables,
        joint_weight,
        10000,
        ETA,
        layered=False,
    )
    time_ldpc += time.perf_counter_ns() - t0
    if cfg.record_intermediate_batches:
        if cfg.try_fix_unreliable_on_last_batch:
            oracle_calls_for_batch.append(pr_oracle.oracle_calls)
            batch_diff = 0
            for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded)):
                actual = np.argmax(actual_pmf) - ETA
                if expect != actual:
                    batch_diff += 1
        differences_for_batch.append(batch_diff)
        differences_for_batches.append(differences_for_batch)
        oracle_calls_for_batches.append(oracle_calls_for_batch)

    # sk_decoded = sk_decoded_marginals
    if cfg.print_intermediate_info or cfg.print_diff:
        matrix = np.zeros((len(all_checks), sk_len), dtype=int)
        for row_idx, check in enumerate(all_checks):
            for col_idx in check:
                matrix[row_idx, col_idx] = 1
        row_counts = np.count_nonzero(matrix, axis=1)
        max_row_weight = np.max(row_counts)
        col_counts = np.count_nonzero(matrix, axis=0)
        max_col_weight = np.max(col_counts)
    if cfg.print_intermediate_info:
        print(f"{max_row_weight=}, {max_col_weight=}")
        intermediate_sk.append(sk_decoded)
        for i, s_i in enumerate(sk):
            pmfs = list(sk_marg[i] for sk_marg in intermediate_sk)
            s_i_idx = s_i + ETA
            is_incorrect = (
                ": WRONG OUTPUT" if np.argmax(intermediate_sk[-1][i]) != s_i_idx else ""
            )
            print(f"{i}: s_i = {s_i}{is_incorrect}, connected checks={col_counts[i]}")
            correct_prs = np.array(list(pmf[s_i_idx] for pmf in pmfs))
            best_guess = np.argmax(correct_prs)
            for j, pmf in enumerate(pmfs):
                selector = "->" if j == best_guess else "  "
                full_ldpc_selector = (
                    " *"
                    if (
                        j == cfg.max_additional_batches
                        or is_in(j, cfg.batch_num_for_full_ldpc)
                    )
                    else ""
                )
                print(f"{selector} {list_small_str(pmf)}{full_ldpc_selector}")

    if cfg.print_diff:
        print(f"key {key_idx}")
    differences = 0
    for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded)):
        actual = np.argmax(actual_pmf) - ETA
        # print(f"{i}: {expect=}, {actual=}, {list_small_str(actual_pmf)}")
        if expect != actual:
            differences += 1
            if cfg.print_intermediate_info or cfg.print_diff:
                print(
                    f"{i}: {expect=}, {actual=}, {list_small_str(actual_pmf)}, checks for var: {col_counts[i]}"
                )
    # lost_info_arr.append(lost_information)
    differences_arr.append(differences)
    total_oracle_calls += pr_oracle.oracle_calls

print(f"Average number of incorrect coefficients = {np.average(differences_arr)}")
print(f"Difference array: {differences_arr}")
# print(f"Lost information: {lost_info_arr}")
print(f"Average oracle calls used = {total_oracle_calls / test_keys}")

if cfg.record_intermediate_batches:
    print(f"Difference data for batches: {differences_for_batches}")
    print(f"Oracle calls data for batches: {oracle_calls_for_batches}")


def time_to_ms_per_key(time_ns, test_keys):
    return float(time_ns / test_keys / 1000000)


print("Timing information (per key) in ms")
print(f"---- conditional pmf computation {time_cond_pr}")
print(f"---- LDPC decoder {time_to_ms_per_key(time_ldpc, test_keys)}")
print(
    f"---- LDPC decoder during batches {time_to_ms_per_key(time_ldpc_batch, test_keys)}"
)
print(
    f"---- Base case handling (everything) {time_to_ms_per_key(time_base, test_keys)}"
)
print(
    f"---- Computation of marginals for base {time_to_ms_per_key(time_base_marginals, test_keys)}"
)
print(
    f"---- Enumerating promising directions {time_to_ms_per_key(time_direction_enumeration, test_keys)}"
)
print(
    f"---- Total time spend in batches {time_to_ms_per_key(time_batches_total, test_keys)}"
)

print(f"{y_statistic=}")
# with open("configuration.data", "wt") as f:
#     for labels, pr_list in sorted(configuration_probs.items(), key=lambda x: x[0]):
#         print(f"configuration {labels} has {pr_list}", file=f)
