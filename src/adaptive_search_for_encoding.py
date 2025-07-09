import itertools as it
from functools import reduce

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from src.kyber_encodings import (
    ETA,
    oracle_threshold,
    potential_z_coefs,
    secret_joint_range,
)


def primitive_tuple(t):
    g = np.gcd.reduce(t)
    return tuple(t_val // g for t_val in t)


# compute the 0-1 mask for dot(z, s) > 0 for all s in {-s_bound, ..., s_bound}^{joint_weight}
# and coefficients of z are first available coefficients (specified by potential_z_coefs)
def precompute_mask_and_directions(first_z, s_bound, joint_weight):
    S = np.array(
        list(secret_joint_range(bound=s_bound, weight=joint_weight)), dtype=int
    )
    small_z_subset = set(potential_z_coefs[:first_z]) | set(
        -x for x in potential_z_coefs[:first_z]
    )
    small_z_subset = list(small_z_subset)
    Z = np.array(
        list(
            set(
                primitive_tuple(z)
                for z in it.product(small_z_subset, repeat=joint_weight)
                if any(z)
            )
        )
    )
    dot = S @ Z.T
    mask = dot > 0
    mask = mask.T

    return mask, Z


def efficient_joint_pmfs(pmfs):
    # return np.einsum("i,j,k,l->ijkl", *pmfs).ravel()
    return reduce(np.multiply.outer, pmfs).ravel()


def most_promising_direction(mask, Z, joint_pmf):
    probs = mask @ joint_pmf
    best_pr_idx = abs(probs - 0.5).argmin()
    best_pr = probs[best_pr_idx]
    best_z = Z[best_pr_idx]
    return best_pr, best_z, mask[best_pr_idx]


def most_promising_two_directions(mask, Z, joint_pmf):
    M = mask * joint_pmf
    probs = M.sum(axis=1)
    best_pr_idx = abs(probs - 0.5).argmin()
    # (greedily) looking for second direction that gives good distribution
    # together with the best one
    best_pr_distr = M[best_pr_idx]
    # leave only probabilities when both encodings give 1; looking at all directions
    p11 = mask @ best_pr_distr
    p10 = (1 - mask) @ best_pr_distr
    p01 = mask @ ((1 - mask[best_pr_idx]) * joint_pmf)
    p00 = 1 - p11 - p10 - p01
    second_best_idx = (
        abs(p00 - 0.25) + abs(p01 - 0.25) + abs(p10 - 0.25) + abs(p11 - 0.25)
    ).argmin()
    best_pr = (
        p00[second_best_idx],
        p01[second_best_idx],
        p10[second_best_idx],
        p11[second_best_idx],
    )
    best_z = (Z[best_pr_idx], Z[second_best_idx])
    return best_pr, best_z, np.column_stack((mask[best_pr_idx], mask[second_best_idx]))


def MILP_create_model(threshold, joint_weight):
    THRESHOLD = threshold
    ABS_Z_BOUND = oracle_threshold // ETA

    joint_size = (2 * ETA + 1) ** joint_weight

    model = gp.Model("halfspace_50_50")
    model.Params.OutputFlag = 1

    z = model.addVars(
        int(joint_weight), lb=-ABS_Z_BOUND, ub=ABS_Z_BOUND, vtype=GRB.INTEGER, name="z"
    )
    allowed_vals = sorted(
        {+c for c in potential_z_coefs} | {-c for c in potential_z_coefs}
    )
    K = len(allowed_vals)
    w = model.addVars(int(joint_weight), K, vtype=GRB.BINARY, name="w")
    for j in range(joint_weight):
        # pick exactly one value
        model.addConstr(w.sum(j, "*") == 1, name=f"pick_one_{j}")
        # link z_j to the chosen value
        model.addConstr(
            z[j] == gp.quicksum(allowed_vals[k] * w[j, k] for k in range(K)),
            name=f"value_match_{j}",
        )

    abs_z = model.addVars(int(joint_weight), vtype=GRB.INTEGER, name="|z|")
    for j in range(joint_weight):
        model.addGenConstrAbs(abs_z[j], z[j])  # |z_j|  =  abs_z[j]
    model.addConstr(abs_z.sum() <= ABS_Z_BOUND)

    # Binary indicator y_j: 1 ↔ dot(z, s_j) > 0.
    y = model.addVars(joint_size, vtype=GRB.BINARY, name="y")

    # Probability that the inequality holds.
    P = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="P")

    # Objective variable d = |P − 0.5|.
    d = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="w")

    # Big‑M linkage constraints.
    for i, s in enumerate(secret_joint_range(ETA, weight=joint_weight)):
        dot = quicksum(z[j] * s[j] for j in range(joint_weight))
        model.addGenConstrIndicator(y[i], True, dot >= THRESHOLD + 1)
        model.addGenConstrIndicator(y[i], False, dot <= THRESHOLD)

    # Linearize absolute deviation.
    model.addConstr(d >= P - float(0.5), name="abs_pos")
    model.addConstr(d >= float(0.5) - P, name="abs_neg")

    # Objective: minimize d.
    model.setObjective(d, GRB.MINIMIZE)

    handles = dict(z=z, y=y, P=P)
    return model, handles


def MILP_run_model(model, handles, joint_pmf):
    y = handles["y"]
    P = handles["P"]
    z = handles["z"]

    # model = model.copy()
    # Define P as the expected value of y.
    model.addConstr(P == quicksum(pr * y[i] for i, pr in enumerate(joint_pmf)))
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        z_opt = [int(round(z[i].X)) for i in range(len(z))]
        model_pr = P.X

        return model_pr, z_opt
    else:
        print("Model did not reach optimality. Status code:", model.Status)
        return None, None


def pmf_mean(pmf):
    return float(np.dot(pmf, range(-ETA, ETA + 1)))


def classify_pmf(
    pmf: np.ndarray,
    certain_thr: float = 0.80,
    dual_thr: float = 0.80,
    dual_certain_thr: float = 0.60,
    entropy_thr: float = 1.00,
    mean_band: float = 0.50,
):
    """
    Return a symbolic label describing the pmf.
    Adjust the four thresholds to taste.
    """
    # ---- basic numbers -------------------------------------------------
    sorted_p = sorted(enumerate(pmf), key=lambda x: x[1], reverse=True)
    max_idx, max_pr = sorted_p[0]
    max_val = max_idx - ETA

    second_idx, second_pr = sorted_p[1]
    second_val = second_idx - ETA

    mean = pmf_mean(pmf)
    # Shannon entropy in bits (ignore zero entries to avoid log(0))
    entropy = -float(sum(p * np.log2(p) for p in pmf if p > 0))
    # entropy = -float((p[p > 0] * np.log2(p[p > 0])).sum())

    # ---- hierarchy of rules -------------------------------------------
    if max_pr >= certain_thr:
        return ("CERT", max_val)

    if max_pr + second_pr >= dual_thr and abs(max_idx - second_idx) == 1:
        if max_pr >= dual_certain_thr:
            return ("DUAL", max_val, second_val)
        else:
            return ("DUAL_UNCERT", max_val, second_val)

    # broad / narrow cut based on entropy
    spread = "NARROW" if entropy < entropy_thr else "BROAD"

    if mean < -mean_band:
        return (spread, "NEG")
    if mean > mean_band:
        return (spread, "POS")
    return (spread, "ZERO")
