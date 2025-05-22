import os
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
    s_distribution_from_hard_y,
)
from simulate_rs import DecoderKyberB2SW2, DecoderKyberB2SW4

from kyber_encodings import (
    ETA,
    anticyclic_shift_multi,
    build_z_values_arr,
    encoding_for_compound_split,
    entropy,
    k,
    n,
    secret_joint_range,
)


def sample_coef_static(expected, pr_oracle):
    return tuple(pr_oracle.predict_bit(val, i) for i, val in enumerate(expected))


def split_from_canonical(z_values, enabled, signs, thresholds, shift):
    z_values = anticyclic_shift_multi(z_values, shift)
    z_values_arr = build_z_values_arr(z_values, enabled, signs)
    return z_values_arr, thresholds


def transform_check_split(configuration, database):
    check_splits_with_idxs = list(zip(*configuration))
    check_splits = []
    for database_idx, shift in check_splits_with_idxs:
        canon_split = database[database_idx]
        z_values, thresholds, enabled, signs, _ = canon_split
        z_values_arr, thresholds = split_from_canonical(
            z_values, enabled, signs, thresholds, shift
        )
        check_splits.append((z_values_arr, thresholds))
    return check_splits


def ldpc_decode(
    check_idxs, secret_variables, check_variables, joint_weight, iterations
):
    num_rows = len(check_idxs)
    n = len(secret_variables)
    matrix = np.zeros((num_rows, n + num_rows), dtype=np.int8)

    for i, indices in enumerate(check_idxs):
        # add 1 for even indices, -1 for odd
        for index in indices:
            matrix[i, index] = 1
        matrix[i, n + i] = -1
    row_counts = np.count_nonzero(matrix, axis=1)
    max_row_weight = np.max(row_counts)
    col_counts = np.count_nonzero(matrix, axis=0)
    max_col_weight = np.max(col_counts)
    if joint_weight == 2:
        decoder_class = DecoderKyberB2SW2
    elif joint_weight == 4:
        decoder_class = DecoderKyberB2SW4
    else:
        raise ValueError(f"{joint_weight} weight for check variable is not supported")
    decoder = decoder_class(
        matrix.astype("int8"), max_col_weight, max_row_weight, iterations
    )
    s_decoded = decoder.decode_with_pr(secret_variables, check_variables)
    return s_decoded


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
        # for block_idx in range(1):
        #     for i in range(3):
        full_rotation_checks.append(
            tuple(x + i + n * block_idx for x in full_rotation_check)
        )
database_4_full = eval(
    open(os.path.join(database_dir, "database_4_full.txt"), "rt").read()
)
full_rotation_split = transform_check_split(
    # ([25, 149, 543, 45, 312, 30, 4], [3, 4, 2, 1, 3, 4, 6]),
    ([523, 201, 465, 25, 149, 543, 45, 312, 30, 4], [1, 5, 0, 3, 4, 2, 1, 3, 4, 6]),
    database_4_full,
)
oracle_configurations.append((full_rotation_checks, full_rotation_split))

# within_block_checks = []
# for block_idx in range(k):
#     for step in [3, 11, 19, 31, 53, 79]:
#         # TODO: end of block is left uncovered
#         steps_total = n // (step * joint_weight)
#         for step_idx in range(steps_total):
#             start_step_block = step_idx * step * joint_weight + n * block_idx
#             for step_offset in range(step):
#                 start = start_step_block + step_offset
#                 check = tuple(start + i * step for i in range(joint_weight))
#                 within_block_checks.append(check)
# database_4_3 = eval(open(os.path.join(database_dir, "database_4_3.txt"), "rt").read())
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
    f"Using {oracle_calls} calls while theory suggest that the minumim is {theoretical_num_of_calls}"
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

    for checks, split in oracle_configurations:
        all_checks.extend(checks)
        encoding_time_start = time.perf_counter()
        check_encoding = []
        for one_bit_split in split:
            z_values_arr, thresholds = one_bit_split
            encoding = encoding_for_compound_split(z_values_arr, thresholds)
            check_encoding.append(encoding)
        check_encoding = list(zip(*check_encoding))
        time_encoding += time.perf_counter() - encoding_time_start
        # TODO: rotate or modify encodings?

        cond_pr_start = time.perf_counter()
        for check_idxs in checks:
            enc_idx = 0
            for var_idx in check_idxs:
                enc_idx = enc_idx * coef_support_size + (sk[var_idx] + ETA)
            x = check_encoding[enc_idx]
            y = sample_coef_static(x, pr_oracle)

            cond_pr = s_distribution_from_hard_y(
                y, pr_oracle, check_encoding, joint_pmf
            )
            # print(f"{cond_pr=}")
            check_variables.append(cond_pr)
        time_cond_pr += time.perf_counter() - cond_pr_start

    ldpc_start = time.perf_counter()
    check_variables = np.array(check_variables, dtype=np.float32)
    sk_decoded = ldpc_decode(
        all_checks, secret_variables, check_variables, joint_weight, 10000
    )
    time_ldpc += time.perf_counter() - ldpc_start
    # print(f"{sk=}")
    # print(f"{sk_decoded=}")
    differences = 0
    for i, (expect, actual_pmf) in enumerate(zip(sk, sk_decoded)):
        actual = np.argmax(actual_pmf) - ETA
        if expect != actual:
            print(f"{i}: {expect=}, {actual=}, {actual_pmf}")
            differences += 1
    differences_arr.append(differences)

print(f"Average number of incorrect coefficients = {np.average(differences_arr)}")

print("Timing information (per key)")
print(f"---- encoding {time_encoding}")
print(f"---- conditional pmf computation {time_cond_pr}")
print(f"---- LDPC decoder {time_ldpc}")
