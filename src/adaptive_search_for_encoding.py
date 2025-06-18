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


def most_promising_direction(mask, Z, pmfs):
    joint_pmf = reduce(np.multiply.outer, pmfs).ravel()
    probs = mask @ joint_pmf
    best_pr, best_z = min(zip(probs, Z), key=lambda x: abs(x[0] - 0.5))
    return best_pr, best_z


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
