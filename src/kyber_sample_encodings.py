import random
from collections import defaultdict

from src.kyber_encodings import (
    B,
    anticyclic_shift,
    build_z_values_arr,
    evaluate_compound_split,
    oracle_threshold,
    potential_z_coefs,
    single_inequality_probability_all_thresholds,
)


def try_generate_z_values(potential_z_coefs, num_variables, max_sum):
    """Generate a valid sample of variables where sum(z) ≤ max_sum."""
    # first put a bound on the sum as this approach tend to generate values with the sum
    # close to the target
    target_sum = random.randint(0, max_sum)
    sampled_values = []
    remaining_sum = target_sum

    for _ in range(num_variables):
        valid_choices = [z for z in potential_z_coefs if z <= remaining_sum]

        z = random.choice(valid_choices)
        sampled_values.append(z)
        remaining_sum -= z

    return sampled_values


def sample_enabled_inequalities_mask(variables):
    if variables not in {2, 4}:
        # TODO: think about how to nicely describe weight distribution
        raise NotImplementedError()

    if variables == 4:
        weight_probabilities = {3: 0.77, 2: 0.13, 1: 0.09, 0: 0.01}
    else:
        weight_probabilities = {1: 0.91, 0: 0.09}

    weights = list(weight_probabilities.keys())
    probabilities = list(weight_probabilities.values())
    chosen_weight = random.choices(weights, probabilities)[0]

    positions = random.sample(range(variables - 1), chosen_weight)
    enabled = [0] * (variables - 1)
    for pos in positions:
        enabled[pos] = 1

    return enabled


def sample_overlay_inequalities(
    pr_lower,
    pr_upper,
    variables,
    u_weight,
    joint_pmf,
    num_iterations,
    use_canonical=True,
):
    max_ineq = variables - u_weight + 1
    comb_overlay_inequality_database = []
    unique_overlay_encodings = defaultdict(set)
    for _ in range(num_iterations):
        z_values = try_generate_z_values(
            potential_z_coefs, u_weight, oracle_threshold // B
        )
        if use_canonical:
            z_values = sorted(z_values)
        z_values = [0] * (variables - u_weight) + z_values
        if not use_canonical:
            random.shuffle(z_values)
        # before considering complex splits, check a single inequality and see what is possible
        probs = single_inequality_probability_all_thresholds(z_values, 4, joint_pmf)
        valid_thresholds = list(k for k, pr in enumerate(probs) if pr > 0)
        if len(valid_thresholds) == 0:
            continue

        # TODO: pretty sure it is enough to consider only threshold_arr with threshold_arr[i] <= threshold_arr[i+1]
        # since we probably can obtain encoding where it is not satisfied by multiplying by -1 when necessary and
        # applying rotation. At least it works with 2 variables, need to check for more variables

        if use_canonical:
            # even if we rotate all max_ineq times, we can't get past pr_lower
            valid_first_thresholds = list(
                k for k, pr in zip(valid_thresholds, probs) if pr * max_ineq >= pr_lower
            )
            if len(valid_first_thresholds) == 0:
                continue
            threshold_k = random.choice(valid_first_thresholds)
            pr = probs[threshold_k]

            thresholds = [threshold_k]
            prev_k = threshold_k
        z_values_arr = [z_values]

        # TODO: I don't know how the encodings will combine, it's hard to estimate when we overshoot pr_upper or
        # undershoot pr_lower. Right now just randomly try, but probably need to try better approach
        enabled = [1] * (max_ineq - 1)
        signs = random.choices(range(2), k=(max_ineq - 1))
        z_shifted = z_values
        for en, sign in zip(enabled, signs):
            z_shifted = anticyclic_shift(z_shifted)
            if not en:
                continue
            z_current = z_shifted
            if sign:
                z_current = list(-x for x in z_current)
            z_values_arr.append(z_current)
            if use_canonical:
                k = random.choice(valid_thresholds[prev_k:])
                thresholds.append(k)
                prev_k = k
        if not use_canonical:
            thresholds = random.choices(valid_thresholds, k=(sum(enabled) + 1))
        pr, encoding = evaluate_compound_split(z_values_arr, thresholds, joint_pmf)
        if pr_lower <= pr <= pr_upper:
            if encoding not in unique_overlay_encodings[pr]:
                unique_overlay_encodings[pr].add(encoding)
                comb_overlay_inequality_database.append(
                    (z_values, thresholds, enabled, signs, pr)
                )
    return sorted(comb_overlay_inequality_database, key=lambda t: abs(t[-1] - 1 / 2))


def sample_full_rotation_inequalities(
    pr_lower,
    pr_upper,
    variables,
    joint_pmf,
    num_iterations,
    use_canonical=True,
):
    comb_inequality_database = []
    for _ in range(num_iterations):
        z_values = try_generate_z_values(
            potential_z_coefs, variables, oracle_threshold // B
        )
        if use_canonical:
            z_values = sorted(z_values)
        # before considering complex splits, check a single inequality and see what is possible
        probs = single_inequality_probability_all_thresholds(z_values, 4, joint_pmf)
        valid_thresholds = list(k for k, pr in enumerate(probs) if pr > 0)
        if len(valid_thresholds) == 0:
            continue

        # TODO: pretty sure it is enough to consider only threshold_arr with threshold_arr[i] <= threshold_arr[i+1]
        # since we probably can obtain encoding where it is not satisfied by multiplying by -1 when necessary and
        # applying rotation. At least it works with 2 variables, need to check for more variables

        if use_canonical:
            # even if we rotate all 'variables' times, we can't get past pr_lower
            valid_first_thresholds = list(
                k
                for k, pr in zip(valid_thresholds, probs)
                if pr * variables >= pr_lower
            )
            if len(valid_first_thresholds) == 0:
                continue
            threshold_k = random.choice(valid_first_thresholds)
            pr = probs[threshold_k]

            thresholds = [threshold_k]

        # TODO: I don't know how the encodings will combine, it's hard to estimate when we overshoot pr_upper or
        # undershoot pr_lower. Right now just randomly try, but probably need to try better approach
        enabled = sample_enabled_inequalities_mask(variables)
        signs = random.choices(range(2), k=(variables - 1))
        z_values_arr = build_z_values_arr(z_values, enabled, signs)
        if use_canonical:
            prev_k = threshold_k
            for en in enabled:
                if not en:
                    continue
                k = random.choice(valid_thresholds[prev_k:])
                thresholds.append(k)
                prev_k = k
        else:
            thresholds = random.choices(valid_thresholds, k=(sum(enabled) + 1))
        pr, encoding = evaluate_compound_split(z_values_arr, thresholds, joint_pmf)
        if pr_lower <= pr <= pr_upper:
            comb_inequality_database.append((z_values, thresholds, enabled, signs, pr))
    return sorted(comb_inequality_database, key=lambda t: abs(t[-1] - 1 / 2))
