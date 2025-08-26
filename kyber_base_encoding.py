import sys
from math import prod

sys.path.append("../SCA-LDPC/simulate-with-python")

import itertools as it
import random

import numpy as np
from simulate.max_likelihood import (
    FalsePositiveNegativePositionalOracle,
    SimpleOracle,
    s_distribution_for_all_y,
)

from src.kyber_encodings import (
    ETA,
    anticyclic_shift,
    anticyclic_shift_multi,
    build_z_values_arr,
    encoding_for_compound_split,
    entropy,
    k,
    n,
    s_joint_to_index,
    secret_distribution,
    secret_joint_prob,
    secret_joint_range,
    split_from_canonical,
)
from src.kyber_sample_encodings import (
    sample_full_rotation_inequalities,
)


def compute_information_of_configuration(
    splits, secret_indices, joint_weight, joint_pmf, pr_oracle, verbose=False
):
    joint_pmf_entropy = entropy(joint_pmf)
    if verbose:
        print(f"Secret entropy of joint: {joint_pmf_entropy}")

    # create a matrix with number of rows equal to the size of truth table over `joint_weight`
    # joint_weight and number of columns equal to number of bits oracle should return
    coding_joint = np.zeros(
        ((2 * ETA + 1) ** joint_weight, sum(len(split) for split in splits)),
        dtype=np.int8,
    )
    coding_offset = 0
    for split, check_indices in zip(splits, secret_indices):
        check_encoding = []
        for one_bit_split in split:
            z_values_arr, thresholds = one_bit_split
            encoding = encoding_for_compound_split(z_values_arr, thresholds)
            check_encoding.append(encoding)
        check_encoding = list(zip(*check_encoding))
        xbits = len(check_encoding[0])
        coding_for_check_dict = {}
        for s, x in zip(secret_joint_range(ETA, weight=joint_weight), check_encoding):
            coding_for_check_dict[s] = x

        for i, s in enumerate(secret_joint_range(ETA, weight=joint_weight)):
            s_subset = tuple(s[i] for i in range(len(s)) if i in check_indices)
            for j in range(xbits):
                coding_joint[i][j + coding_offset] = coding_for_check_dict[s_subset][j]
        coding_offset += xbits
    # print_coding(coding_joint, weight=joint_weight)

    # s_distribution_for_all_y(pr_oracle, coding, s_pmf_array)
    cond_prob_all, pr_of_y = s_distribution_for_all_y(
        pr_oracle,
        coding_joint,
        joint_pmf,
    )
    # print(cond_prob_all)
    if verbose:
        print(f"{pr_of_y=}; entropy is {entropy(pr_of_y)}")
    # print(f"H(Y) - 2 H(p) = {entropy(pr_of_y) - 2 * (1 - bsc_entropy)}")

    expected_info = 0
    for i, y in enumerate(it.product(range(2), repeat=len(coding_joint[0]))):
        if pr_of_y[i] == 0:
            continue
        info = joint_pmf_entropy - entropy(cond_prob_all[i])
        if verbose:
            print(
                f"Information on {y} for joint is {info}, probability to get y is {pr_of_y[i]}"
            )
        expected_info += info * pr_of_y[i]

    return expected_info


# as input take tuples with indices from database with shifts, return actual entries
def transform_splits(splits):
    splits_transformed = []
    for check_splits in splits:
        transformed_check = []
        for split in check_splits:
            canon_split, shift = split
            z_values, thresholds, enabled, signs, _ = canon_split
            z_values_arr, thresholds = split_from_canonical(
                z_values, enabled, signs, thresholds, shift
            )
            transformed_check.append((z_values_arr, thresholds))
        splits_transformed.append(transformed_check)
    return splits_transformed


def greedy_sampling_single(tests_num, oracle_calls, database, joint_weight):
    assert oracle_calls >= 2
    id_data_best = [[], []]
    best_info = None
    for j in range(2, oracle_calls + 1):
        take_from_database = 1 if j > 2 else 2
        info_configurations = []
        for i in range(tests_num):
            splits = []
            database_idxs = random.choices(range(len(database)), k=take_from_database)
            shifts = random.choices(range(2 * joint_weight), k=take_from_database)
            database_idxs.extend(id_data_best[0])
            shifts.extend(id_data_best[1])
            for idx, shift in zip(database_idxs, shifts):
                database_entry = database[idx]
                splits.append((database_entry, shift))
            splits = [splits]
            splits_transformed = transform_splits(splits)

            expected_info = compute_information_of_configuration(
                splits_transformed,
                secret_indices,
                joint_weight,
                joint_pmf,
                pr_oracle,
                verbose=False,
            )
            info_configurations.append((expected_info, (database_idxs, shifts)))
        best_configuration = max(info_configurations, key=lambda x: x[0])
        id_data_best = best_configuration[1]
        best_info = best_configuration[0]
        num_calls = len(id_data_best[0])
        theoretical_limit = (1 - bsc_entropy) * num_calls
        print(
            f"Result so far ({num_calls} calls): {best_info=:.5f}, theory={theoretical_limit:.5f}, diff={(theoretical_limit - best_info):.5f}, {id_data_best=}"
        )
    return best_info, id_data_best


s_distr = secret_distribution(ETA)
s_entropy = entropy(s_distr)
print(f"Secret entropy: {s_entropy}")
print(
    f"Approximated entropy of single coefficient = {s_entropy}, in total = {(s_entropy * n * k)}"
)

joint_weight = 4
joint_pmf = list()
for s in secret_joint_range(ETA, weight=joint_weight):
    joint_pmf.append(prod(s_distr[si] for si in s))
secret_indices = [list(range(joint_weight))]

p = 0.95
pr_oracle = SimpleOracle(p)
bsc_entropy = entropy([p, 1 - p])

pr_lower, pr_upper = 0.48, 0.52
inequality_database = sample_full_rotation_inequalities(
    pr_lower,
    pr_upper,
    joint_weight,
    joint_pmf,
    10000,
    use_canonical=True,
)
with open("tmp_database.txt", "wt") as f:
    print(inequality_database, file=f)

# inequality_database = eval(open("tmp_database.txt", "rt").read())

tests_num = 10000
best_info, id_data_best = greedy_sampling_single(
    tests_num, 8, inequality_database, joint_weight
)
print(f"{best_info=}, {id_data_best=}")
