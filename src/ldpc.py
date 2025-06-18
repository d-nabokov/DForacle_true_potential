import numpy as np
from simulate_rs import DecoderKyberB2SW2, DecoderKyberB2SW4


def sample_coef_static(expected, pr_oracle):
    return tuple(pr_oracle.predict_bit(val, i) for i, val in enumerate(expected))


def bit_tuple_to_int(t):
    res = 0
    for bit in t:
        res = (res << 1) | bit
    return res


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
