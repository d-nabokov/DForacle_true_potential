import os
import pickle
import random
import sys
import time
from collections import defaultdict
from math import comb, log, prod

import numpy as np

sys.path.append("../SCA-LDPC/simulate-with-python")

from simulate.kyber import sample_secret_coefs, secret_distribution
from simulate.max_likelihood import (
    SimpleOracle,
    s_distribution_for_all_y,
    s_distribution_from_hard_y,
)
from simulate_rs import DecoderKyberB2SW2, DecoderKyberB2SW4

from src.adaptive_search_for_encoding import (
    MILP_create_model,
    MILP_run_model,
    most_promising_direction,
    precompute_mask_and_directions,
)
from src.kyber_encodings import (
    ETA,
    anticyclic_shift_multi,
    build_z_values_arr,
    encoding_for_compound_split,
    entropy,
    k,
    marginal_pmf,
    marginal_pmfs,
    multibit_encoding,
    n,
    secret_joint_range,
    split_from_canonical,
    transform_check_split,
)
from src.ldpc import bit_tuple_to_int, ldpc_decode, sample_coef_static


def list_small_str(lst, precision=3):
    lst_str = ", ".join(f"{val:.{precision}f}" for val in lst)
    return "[" + lst_str + "]"


USE_DYNAMIC_ADDITIONAL = True

p = 0.95
pr_oracle = SimpleOracle(p)

sk_len = n * k

joint_weight = 4
prob_s = secret_distribution()
joint_pmf = list()
for s in secret_joint_range(ETA, weight=joint_weight):
    joint_pmf.append(prod(prob_s[si] for si in s))

s_prior = list(list(prob_s.values()) for _ in range(sk_len))
secret_variables = np.array(s_prior, dtype=np.float32)


database_dir = "database_10000_ordered"

oracle_configurations = []

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
full_rotation_split = transform_check_split(
    ([25, 149, 543, 45, 312, 30, 4], [3, 4, 2, 1, 3, 4, 6]),
    # ([523, 201, 465, 25, 149, 543, 45, 312, 30, 4], [1, 5, 0, 3, 4, 2, 1, 3, 4, 6]),
    database_4_full,
)
oracle_configurations.append((full_rotation_checks, full_rotation_split))
if USE_DYNAMIC_ADDITIONAL:
    with open("best_conditional_full_rotation", "rb") as f:
        best_conditional_full_rotation_checks = pickle.load(f)

# with open("cond_prob_all_y_095_7bits", "rb") as f:
#     all_y_pmf, pr_y = pickle.load(f)
# for y_idx, y_pmf in enumerate(all_y_pmf):
#     s_marginal = marginal_pmf(y_pmf, joint_weight)
#     best_check = best_conditional_full_rotation_checks[y_idx]
#     print(",".join(map(list_small_str, s_marginal)), best_check)
# # print(best_conditional_full_rotation_checks[0])
# exit()

within_block_checks = []
for block_idx in range(k):
    for step in [3, 11, 19]:
        # TODO: end of block is left uncovered
        steps_total = n // (step * joint_weight)
        for step_idx in range(steps_total):
            start_step_block = step_idx * step * joint_weight + n * block_idx
            for step_offset in range(step):
                start = start_step_block + step_offset
                check = tuple(start + i * step for i in range(joint_weight))
                within_block_checks.append(check)
# database_4_3 = eval(
#     open(os.path.join("database_conditional", "database_4_3.txt"), "rt").read()
# )
# # based on (1, 4, 7, 10) check from one sk after base 7 calls
# database_4_3 = [([10, 13, 0, 59], [0, 0], [1], [0], 0.48366924219397456)]
# # based on (2, 5, 8, 11) check from one sk after base 7 calls
# database_4_3 = [([23, 39, 130, 0], [1, 0], [1], [1], 0.4995468324008786)]
# within_block_split = transform_check_split(
#     ([0], [0]),
#     database_4_3,
# )
# within_block_split = [None, None, None, None, None, None]
# oracle_configurations.append((within_block_checks, within_block_split))

# between_block_checks = []
# # for block_idx in range(k):
# #     # for step in [3, 11, 19, 31, 53, 79]:
# #     for step in [3, 11, 31]:
# #         # TODO: end of block is left uncovered
# #         steps_total = n // (step * joint_weight)
# #         for step_idx in range(steps_total):
# #             start_step_block = step_idx * step * joint_weight + n * block_idx
# #             for step_offset in range(step):
# #                 start = start_step_block + step_offset
# #                 check = tuple(start + i * step for i in range(joint_weight))
# #                 within_block_checks.append(check)
# database_4_4 = eval(open(os.path.join(database_dir, "database_4_3.txt"), "rt").read())
# within_block_split = transform_check_split(
#     ([440, 202, 82], [3, 0, 3]),
#     database_4_3,
# )
# oracle_configurations.append((within_block_checks, within_block_split))

oracle_calls = 0
for checks, split in oracle_configurations:
    oracle_calls += len(checks) * len(split)
max_info_per_call = 1 - entropy([p, 1 - p])
theoretical_num_of_calls = n * k * entropy(prob_s) / max_info_per_call
print(
    f"Using {oracle_calls} calls while theory suggest that the minimum is {theoretical_num_of_calls}"
)

coef_support_size = len(prob_s)
random.seed(42)
differences_arr = []

time_encoding = 0
time_cond_pr = 0
time_ldpc = 0
test_keys = 1
for key_idx in range(test_keys):
    sk = sample_secret_coefs(sk_len)
    all_checks = []
    check_variables = []
    oracle_calls = 0

    sk_decoded_marginals = [0] * sk_len

    # base case: full rotation checks
    checks, split = full_rotation_checks, full_rotation_split
    check_encoding = multibit_encoding(split)
    if p == 0.95 and len(split) == 7:
        with open("cond_prob_all_y_095_7bits", "rb") as f:
            all_y_pmf, pr_y = pickle.load(f)
    else:
        all_y_pmf, pr_y = s_distribution_for_all_y(pr_oracle, check_encoding, joint_pmf)

    all_checks.extend(checks)
    for check_idxs in checks:
        enc_idx = 0
        for var_idx in check_idxs:
            enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
        x = check_encoding[enc_idx]
        y = sample_coef_static(x, pr_oracle)
        y_idx = bit_tuple_to_int(y)
        pmf = all_y_pmf[y_idx]

        if USE_DYNAMIC_ADDITIONAL:
            # dynamically adding new encoding based on current y
            z_values, thresholds, enabled, signs, _ = (
                best_conditional_full_rotation_checks[y_idx]
            )
            z_values_arr, thresholds = split_from_canonical(
                z_values, enabled, signs, thresholds, 0
            )
            additional_encoding = multibit_encoding([(z_values_arr, thresholds)])
            additional_x = additional_encoding[enc_idx]
            additional_y = sample_coef_static(additional_x, pr_oracle)

            new_pmf = s_distribution_from_hard_y(
                additional_y,
                pr_oracle,
                additional_encoding,
                pmf,
            )
            pmf = new_pmf
        check_variables.append(pmf)
        s_marginal = marginal_pmf(pmf, joint_weight)
        for i, var_idx in enumerate(check_idxs):
            sk_decoded_marginals[var_idx] = s_marginal[i]

    # for checks, split in oracle_configurations:
    #     all_checks.extend(checks)
    #     encoding_time_start = time.perf_counter()
    #     check_encoding = []
    #     for one_bit_split in split:
    #         z_values_arr, thresholds = one_bit_split
    #         encoding = encoding_for_compound_split(z_values_arr, thresholds)
    #         check_encoding.append(encoding)
    #     check_encoding = list(zip(*check_encoding))
    #     time_encoding += time.perf_counter() - encoding_time_start
    #     # TODO: rotate or modify encodings?

    #     if p == 0.95 and len(split) == 7:
    #         with open("cond_prob_all_y_095_7bits", "rb") as f:
    #             all_y_pmf, pr_y = pickle.load(f)
    #     else:
    #         all_y_pmf, pr_y = s_distribution_for_all_y(
    #             pr_oracle, check_encoding, joint_pmf
    #         )
    #     s_marginals = marginal_pmfs(all_y_pmf, joint_weight)

    #     cond_pr_start = time.perf_counter()
    #     for check_idxs in checks:
    #         enc_idx = 0
    #         for var_idx in check_idxs:
    #             enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
    #         x = check_encoding[enc_idx]
    #         y = sample_coef_static(x, pr_oracle)
    #         y_idx = bit_tuple_to_int(y)
    #         cond_pr = all_y_pmf[y_idx]
    #         check_variables.append(cond_pr)
    #         for i, var_idx in enumerate(check_idxs):
    #             sk_decoded_marginals[var_idx] = s_marginals[y_idx][i]
    #     time_cond_pr += time.perf_counter() - cond_pr_start

    with open("sk_marginals", "wb") as f:
        pickle.dump(sk_decoded_marginals, f)

    ldpc_start = time.perf_counter()
    check_variables = np.array(check_variables, dtype=np.float32)
    sk_decoded = ldpc_decode(
        all_checks, secret_variables, check_variables, joint_weight, 10000
    )
    time_ldpc += time.perf_counter() - ldpc_start
    # print(f"{sk=}")
    # print(f"{sk_decoded=}")

    def list_small_str(lst, precision=3):
        lst_str = ", ".join(f"{val:.{precision}f}" for val in lst)
        return "[" + lst_str + "]"

    # sk_decoded = sk_decoded_marginals

    differences = 0
    for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded)):
        actual = np.argmax(actual_pmf) - ETA
        if expect != actual:
            differences += 1
            # print(
            #     f"{i}: {expect=}, {actual=}, {list_small_str(actual_pmf)}, {list_small_str(sk_decoded_marginals[i])}"
            # )
    differences_arr.append(differences)

print(f"Average number of incorrect coefficients = {np.average(differences_arr)}")

print("Timing information (per key)")
print(f"---- encoding {time_encoding}")
print(f"---- conditional pmf computation {time_cond_pr}")
print(f"---- LDPC decoder {time_ldpc}")
