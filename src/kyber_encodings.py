import itertools as it
from collections import defaultdict
from math import comb, log, prod

import numpy as np

KYBER_K = 2

if KYBER_K == 2:
    # Kyber-512 params
    Q = 3329
    ETA = 3
    block_len = 256
    num_blocks = 2
    du = 10
    dv = 4
    SMALLEST_THRESHOLD = 208
elif KYBER_K == 3:
    # Kyber-768 params
    Q = 3329
    ETA = 2
    block_len = 256
    num_blocks = 3
    du = 10
    dv = 4
    SMALLEST_THRESHOLD = 208
elif KYBER_K == 4:
    # Kyber-1024 params
    Q = 3329
    ETA = 2
    block_len = 256
    num_blocks = 4
    du = 11
    dv = 5
    SMALLEST_THRESHOLD = 104
else:
    raise ValueError()


n = block_len
k = num_blocks


# replacement for SageMath's binomial
def binomial(n, k):
    return comb(n, k)


def entropy(distr):
    if type(distr) is dict or type(distr) is defaultdict:
        return -sum(p * log(p, 2) for p in distr.values() if p > 0)
    else:
        return -sum(p * log(p, 2) for p in distr if p > 0)


def secret_distribution(eta, weight=1):
    B = weight * eta
    n = 2 * B
    den = 2**n
    return {s: (binomial(n, s + B) / den) for s in range(-B, B + 1)}


def sum_secret_distr(distr, weight):
    B = (len(distr) - 1) // 2
    BSUM = B * weight
    d = defaultdict(float)
    for values in it.product(range(-B, B + 1), repeat=weight):
        d[sum(values)] += prod(distr[val] for val in values)
    return d


def secret_range(sum_weight):
    return range(-sum_weight, sum_weight + 1)


def secret_range_len(sum_weight):
    return 2 * sum_weight + 1


def coding_from_patterns(pattern, sum_weight=1):
    B = sum_weight
    if len(pattern) != (2 * B + 1):
        raise ValueError("len of pattern doesn't match sum weight")
    if type(pattern[0]) is tuple:
        return {s: p_val for s, p_val in zip(range(-B, B + 1), pattern)}
    else:
        return {s: (p_val,) for s, p_val in zip(range(-B, B + 1), pattern)}


# s_distr = secret_distribution(ETA)
# secret_coef_entropy = entropy(s_distr)

# print(
#     f"Approximated entropy of single coefficient = {secret_coef_entropy}, in total = {(secret_coef_entropy * n * k)}"
# )

# sum_weight = 4
# beta_distr = list(sum_secret_distr(s_distr, i + 1) for i in range(sum_weight))
# for i, distr in enumerate(beta_distr):
#     print(f"weight = {i + 1}, entropy = {float(entropy(distr)):.3f}")
#     print(", ".join(f"{float(x):.4f}" for x in distr.values()))


def compress(x, d):
    val = (2**d / Q) * x
    return round(val) % (2**d)


def decompress(x, d):
    return round((Q / 2**d) * x)


def center(x):
    """
    Map each entry of x into the symmetric range (-q/2, q/2].
    """
    return (x + Q // 2) % Q - Q // 2


v_values = {}
vprimes = set()
vprimeprimes = set()
for v in range(Q):
    vprime = compress(v, dv)
    vprimes.add(vprime)
    vprimeprime = decompress(vprime, dv)
    vprimeprimes.add(vprimeprime)
    v_values[v] = (vprime, vprimeprime)

u_values = {}
uprimes = set()
uprimeprimes = set()
for u in range(Q):
    uprime = compress(u, du)
    uprimes.add(uprime)
    uprimeprime = decompress(uprime, du)
    uprimeprimes.add(uprimeprime)
    u_values[u] = (uprime, uprimeprime)


# prob_s = beta_distr[0]
# s_range = np.array(list(prob_s.keys()))
# p_values = np.array(list(prob_s.values()))

# B = len(s_range) // 2
B = ETA
oracle_threshold = Q // 4
Z_BOUND = oracle_threshold // ETA
k_step = SMALLEST_THRESHOLD
potential_z_coefs = sorted(z for z in uprimeprimes if z * B <= oracle_threshold)


# Anticyclic aka negacyclic shift
def anticyclic_shift(z):
    return [
        -z[-1],
    ] + list(z[:-1])


def anticyclic_shift_multi(z, steps):
    lz = len(z)
    steps %= 2 * lz
    if steps < lz:
        sign = 1
    else:
        sign = -1
        steps %= lz
    return list(-sign * x for x in z[lz - steps :]) + list(
        sign * x for x in z[: lz - steps]
    )


def secret_joint_range(bound, weight):
    for s in it.product(range(-bound, bound + 1), repeat=weight):
        yield s


def s_joint_to_index(s, bound):
    res = 0
    mult = 1
    for s_val in s:
        res += (s_val + bound) * mult
        mult *= 2 * bound + 1
    return res


def secret_joint_prob(s, s_distr):
    return prod(s_distr[s_val] for s_val in s)


def evaluate_pr_of_compound_split(z_values_arr, threshold_values, joint_pr):
    """
    z_values_arr: list of list of coefficients for z.
    threshold_values: list of thresholds, where each threshold is k, not k * k_step.
    joint_pr: list of probabilities.
    """
    true_prob = 0
    for s, s_pr in zip(secret_joint_range(ETA, weight=len(z_values_arr[0])), joint_pr):
        q = False
        for z_values, threshold in zip(z_values_arr, threshold_values):
            uv = np.dot(z_values, s)
            q = q or (uv > (threshold * k_step))
        if q:
            true_prob += s_pr

    return true_prob


def evaluate_compound_split(z_values_arr, threshold_values, joint_pr):
    """
    z_values_arr: list of list of coefficients for z.
    threshold_values: list of thresholds, where each threshold is k, not k * k_step.
    joint_pr: list of probabilities.
    """
    encoding = []
    true_prob = 0
    for s, s_pr in zip(secret_joint_range(ETA, weight=len(z_values_arr[0])), joint_pr):
        q = False
        for z_values, threshold in zip(z_values_arr, threshold_values):
            uv = np.dot(z_values, s)
            q = q or (uv > (threshold * k_step))
        encoding.append(int(q))
        if q:
            true_prob += s_pr

    return true_prob, tuple(encoding)


def encoding_for_compound_split(z_values_arr, threshold_values):
    """
    z_values_arr: list of list of coefficients for z.
    threshold_values: list of thresholds, where each threshold is k, not k * k_step.
    joint_pr: list of probabilities.
    """
    encoding = []
    for s in secret_joint_range(ETA, weight=len(z_values_arr[0])):
        q = False
        for z_values, threshold in zip(z_values_arr, threshold_values):
            uv = np.dot(z_values, s)
            q = q or (uv > (threshold * k_step))

        encoding.append(int(q))

    return tuple(encoding)


def multibit_encoding(split):
    check_encoding = []
    for one_bit_split in split:
        z_values_arr, thresholds = one_bit_split
        encoding = encoding_for_compound_split(z_values_arr, thresholds)
        check_encoding.append(encoding)
    return list(zip(*check_encoding))


def single_inequality_probability_all_thresholds(z_values, max_threshold, joint_pr):
    """
    Compute probability of (z_values * s) > threshold for threshold in range(max_threshold)
    """
    true_prob = [0] * max_threshold
    real_thresholds = list(threshold * k_step for threshold in range(max_threshold))
    for s, s_pr in zip(secret_joint_range(ETA, weight=len(z_values)), joint_pr):
        uv = np.dot(z_values, s)
        for i, real_threshold in enumerate(real_thresholds):
            if uv > real_threshold:
                true_prob[i] += s_pr
    return true_prob


def single_inequality_pr(z_values, threshold, joint_pmf):
    true_prob = 0
    real_threshold = threshold * k_step
    for s, s_pr in zip(secret_joint_range(ETA, weight=len(z_values)), joint_pmf):
        uv = np.dot(z_values, s)
        if uv > real_threshold:
            true_prob += s_pr
    return true_prob


def build_z_values_arr(z_values, enabled, signs):
    z_values_arr = [z_values]

    z_shifted = z_values
    for en, sign in zip(enabled, signs):
        z_shifted = anticyclic_shift(z_shifted)
        if not en:
            continue
        z_current = z_shifted
        if sign:
            z_current = list(-x for x in z_current)
        z_values_arr.append(z_current)
    return z_values_arr


def split_from_canonical(z_values, enabled, signs, thresholds, shift):
    z_values = anticyclic_shift_multi(z_values, shift)
    z_values_arr = build_z_values_arr(z_values, enabled, signs)
    return z_values_arr, thresholds


def inequalities_to_split(inequalities):
    check_splits = []
    for z_values, thresholds, enabled, signs in inequalities:
        z_values_arr = build_z_values_arr(z_values, enabled, signs)
        check_splits.append((z_values_arr, thresholds))
    return check_splits


def get_inequalities(configuration, database):
    check_splits_with_idxs = list(zip(*configuration))
    inequalities = []
    for database_idx, shift in check_splits_with_idxs:
        canon_split = database[database_idx]
        z_values, thresholds, enabled, signs, _ = canon_split
        z_values = anticyclic_shift_multi(z_values, shift)
        inequalities.append((z_values, thresholds, enabled, signs))
    return inequalities


# from a list of a joint pmf for each possible y compute a list of marginal pmfs for each variable
def marginal_pmfs(cond_prob_all, weight):
    y_len = len(cond_prob_all)
    s_marj_cond = np.zeros((y_len, weight, (2 * ETA + 1)), dtype=np.float32)
    for y in range(y_len):
        for j, s in enumerate(secret_joint_range(ETA, weight=weight)):
            for s_idx in range(weight):
                s_val = s[s_idx]
                s_marj_cond[y][s_idx][s_val + ETA] += cond_prob_all[y][j]
    return s_marj_cond


def marginal_pmf(pmf, weight):
    s_marj_cond = np.zeros((weight, (2 * ETA + 1)), dtype=np.float32)
    for j, s in enumerate(secret_joint_range(ETA, weight=weight)):
        for s_idx in range(weight):
            s_val = s[s_idx]
            s_marj_cond[s_idx][s_val + ETA] += pmf[j]
    return s_marj_cond


def rearrange_tuple(original_tuple, n, variables, reverse=None, permute=None):
    """
    Rearranges a flattened tuple based on reversing and permuting axes.

    Parameters:
    - original_tuple: The original tuple (flattened).
    - n: The size of each dimension (all dimensions are (n, ..., n)).
    - variables: Number of dimensions (i.e., length of shape).
    - reverse: A tuple of booleans indicating which axes to reverse (default: None -> no reversals).
    - permute: A tuple specifying a new order for the axes (default: None -> no permutation).

    Returns:
    - A new tuple with elements rearranged according to the reversed and permuted axes.
    """
    shape = (n,) * variables  # Define the shape dynamically

    if reverse is None:
        reverse = (False,) * variables  # Default: No reversals

    if permute is None:
        permute = tuple(range(variables))  # Default: No reordering

    # Convert to Numpy array with given shape
    array = np.array(original_tuple).reshape(shape)

    # Apply reversals
    for axis, reverse in enumerate(reverse):
        if reverse:
            array = np.flip(array, axis=axis)

    # Apply permutation of axes
    array = np.transpose(array, permute)

    # Flatten back to tuple
    return tuple(array.flatten())
