import itertools as it
import random

import numpy as np
from simulate_rs import DecoderKyberB2SW2, DecoderKyberB2SW4, DecoderKyberB3SW4


def sample_coef_static(expected, pr_oracle):
    return tuple(pr_oracle.predict_bit(val, i) for i, val in enumerate(expected))


def bit_tuple_to_int(t):
    res = 0
    for bit in t:
        res = (res << 1) | bit
    return res


def compute_pairs(checks):
    """
    Given an iterable of parity-check rows (each a sequence of sorted column indices
    where the entry is 1), return the **set of all unordered column pairs**
    that ever appear together in a row.
    """
    pairs = set()
    for row in checks:
        for a, b in it.combinations(row, 2):
            pairs.add((a, b))
    return pairs


def generate_ldpc_block(
    seen_pairs,
    n_rows=192,
    row_weight=4,
    rng=random,
    max_attempts=2_000,
):
    """
    Create a block of `n_rows` rows for an LDPC parity-check sub-matrix with
    the constraints

        • row weight           = `row_weight`         (default 4)
        • each column used once inside the block
        • **no length-4 cycles:** no column pair may
          have been seen before in `seen_pairs`.
    """
    n_cols = n_rows * row_weight

    for _ in range(max_attempts):
        cols = list(range(n_cols))
        rng.shuffle(cols)

        # initial candidate list (fixed column-disjoint property)
        cand_rows = [
            tuple(sorted(cols[i * row_weight : (i + 1) * row_weight]))
            for i in range(n_rows)
        ]

        used_pairs = set(seen_pairs)  # grow as we accept rows
        block = []  # accepted rows
        cursor = 0  # current index inside cand_rows

        # We’ll keep looping while we make progress; a “stall” ends the attempt
        progress_made = True
        while progress_made and len(block) < n_rows:
            progress_made = False

            while cursor < len(cand_rows):
                row = cand_rows[cursor]
                row_pairs = {(a, b) for a, b in it.combinations(row, 2)}

                # accept or defer
                if row_pairs.isdisjoint(used_pairs):
                    block.append(row)
                    used_pairs.update(row_pairs)
                    cursor += 1
                    progress_made = True  # we placed something
                else:
                    # push conflicting row to the end & reshuffle the unseen tail
                    tail = cand_rows[cursor + 1 :]
                    rng.shuffle(tail)
                    cand_rows[cursor + 1 :] = tail
                    cand_rows.append(cand_rows.pop(cursor))
                    # cursor stays on the *new* row that just moved into its position

            # if cursor ran to the end without filling the block, we stalled
        if len(block) == n_rows:
            return block, used_pairs

        # otherwise: failed attempt → try a fresh shuffle
    raise RuntimeError(
        f"Unable to build a conflict-free block after {max_attempts} attempts."
    )


# Very naive implementation, not very efficient, but doesn't matter
def generate_ldpc_matrix(n, check_weight, num_blocks, seen_checks, rng=random):
    new_checks = []
    allowed_idxs = []
    for _ in range(n):
        allowed_idxs.append(set(range(n)))
    for check in seen_checks:
        for a, b in it.combinations(check, 2):
            allowed_idxs[a].discard(b)
            allowed_idxs[b].discard(a)
    for block_no in range(num_blocks):
        block_variables_to_choose = set(range(n))
        for check_no in range(n // check_weight):
            new_var = block_variables_to_choose.pop()
            allowed_for_check = allowed_idxs[new_var]
            new_check = [new_var]
            for i in range(1, check_weight):
                # TODO: bad code, shouldn't sample from set
                new_var = rng.sample(allowed_for_check, 1)[0]
                block_variables_to_choose.discard(new_var)
                new_check.append(new_var)
                if i != check_weight - 1:
                    allowed_for_check = allowed_for_check.intersection(
                        allowed_idxs[new_var]
                    )
                    if len(allowed_for_check) == 0:
                        print(
                            f"Tried getting {new_check} check, but no available variables left"
                        )
                        return None
            new_checks.append(sorted(new_check))
            for a, b in it.combinations(new_check, 2):
                allowed_idxs[a].discard(b)
                allowed_idxs[b].discard(a)
    return new_checks


def ldpc_decode(
    check_idxs,
    secret_variables,
    check_variables,
    joint_weight,
    iterations,
    eta,
    layered=False,
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
    decoder_map = {
        (2, 2): DecoderKyberB2SW2,
        (4, 2): DecoderKyberB2SW4,
        (4, 3): DecoderKyberB3SW4,
    }
    decoder_class = decoder_map.get((joint_weight, eta))
    if decoder_class is None:
        raise ValueError(
            f"Configuration with weight={joint_weight},eta={eta} is not supported"
        )
    decoder = decoder_class(
        matrix.astype("int8"), max_col_weight, max_row_weight, iterations
    )
    if layered:
        s_decoded = decoder.decode_with_pr_layered(secret_variables, check_variables)
    else:
        s_decoded = decoder.decode_with_pr(secret_variables, check_variables)
    return s_decoded
