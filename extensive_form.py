import os
import time
import sys

import itertools
import numpy as np
import random
from scipy.stats import norm

# Get the directory containing readSMPS
readsmps_dir = "/Users/jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS"

# Add it to sys.path
if readsmps_dir not in sys.path:
    sys.path.append(readsmps_dir)

from readSMPS.decmps_2slp import RandVars
from readSMPS.decmps_2slp import decompose


def get_obs_probs(rvs, sampling=False, iterations=None):
    print("The number of random variables equals:", rvs.rvnum)
    if not sampling:
        # After processing the stochastic data, compute observations and probabilities
        rand_vars_values = [list(rvs.dist[i].keys()) for i in range(rvs.rvnum)]
        observations = list(itertools.product(*rand_vars_values))

        rand_vars_probs = [list(rvs.dist[i].values()) for i in range(rvs.rvnum)]
        combinations = itertools.product(*rand_vars_probs)
        probabilities = [np.prod(combination) for combination in combinations]

        print("The number of scenarios equals:", len(observations))

    else:
        observations = []

        for _ in range(iterations):
            sample_tuple = []
            for rv in rvs.dist:
                outcomes, probs = zip(*rv.items())  # Extract outcomes and probabilities
                sampled_outcome = random.choices(outcomes, probs)[
                    0
                ]  # Sample based on probabilities
                sample_tuple.append(sampled_outcome)
            observations.append(tuple(sample_tuple))

        probabilities = [
            1 / iterations
        ] * iterations  # Equal probability for each sampled observation

    return observations, probabilities


def extensive_form(
    model,
    sampling=False,
    iterations=None,
    replication=0,
    upper_bound=False,
    xhat=None,
    method="extensive_form",
):
    rand_vars = RandVars(model.name)

    start_get_obs_probs = time.time()
    observations, probabilities = get_obs_probs(rand_vars, sampling, iterations)
    end_get_obs_probs = time.time()
    elapsed_get_obs_probs = end_get_obs_probs - start_get_obs_probs
    print(
        f"Time taken to enumerate over all random variables: {elapsed_get_obs_probs:.6f} seconds."
    )

    model.create_master(method)

    if not upper_bound:
        start_create_ef = time.time()

        for iteration, obs in enumerate(observations, start=1):
            prob = probabilities[iteration - 1]
            model.create_sub(obs, prob, iteration)
        end_create_ef = time.time()
        elapsed_create_ef = end_create_ef - start_create_ef
        print(
            f"Time taken to build the extensive form: {elapsed_create_ef:.6f} seconds."
        )
        up_bound = None

    if upper_bound and replication > 0:
        variables = model.prob.master_model.getVars()
        cost_coeff = [var.getAttr("Obj") for var in variables]

        recourse_val_list = []
        for iteration, obs in enumerate(observations, start=1):
            prob = probabilities[iteration - 1]
            model.create_sub(obs, prob, iteration)

            model.create_sub(obs, xhat, iteration)
            model.prob.sub_model.setParam("OutputFlag", 0)
            model.prob.sub_model.optimize()
            recourse_val = model.prob.sub_model.objVal
            recourse_val_list.append(recourse_val)

        mean_recourse_val = sum(recourse_val_list) / len(recourse_val_list)

        up_bound = np.dot(cost_coeff, xhat) + mean_recourse_val

    # Solve the extensive form
    model.prob.master_model.setParam("OutputFlag", 0)
    start_optimize_ef = time.time()
    model.prob.master_model.optimize()
    end_optimize_ef = time.time()
    elapsed_optimize_ef = end_optimize_ef - start_optimize_ef
    print(f"Time taken to solve the extensive form: {elapsed_optimize_ef:.6f} seconds.")

    obj_value = model.prob.master_model.ObjVal

    # Retrieve first-stage variable values
    first_stage_sol = {}
    for v in model.prob.master_model.getVars()[: model.prob.master_var_size]:
        first_stage_sol[v.VarName] = v.X

    return obj_value, first_stage_sol, up_bound


def confidence_interval(value_list, alpha=0.05):
    M = len(value_list)

    if M == 1:
        return value_list[0], None

    mean = sum(value_list) / len(value_list)
    diff = [(v - mean) for v in value_list]
    sum_sqr_diff = sum([d**2 for d in diff])

    sample_var_est = sum_sqr_diff / (M - 1)

    beta = alpha / 2
    z_beta = norm.ppf(1 - beta)

    lo_bnd = mean - z_beta * sample_var_est / np.sqrt(M)
    up_bnd = mean + z_beta * sample_var_est / np.sqrt(M)

    conf_interval = (lo_bnd, up_bnd)

    return mean, conf_interval


def main():
    start_main = time.time()

    instance = "lands3"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    sampling = True

    # Set parameters for the case sampling = True
    iterations = 10000
    replications = 1
    upper_bound = False

    d = decompose(f"{instance}", input_dir)
    d.find_stage_idx()
    method = "extensive_form"

    if not sampling:
        obj_value, first_stage_sol, up_bound = extensive_form(
            d,
            sampling,
            iterations=None,
            replication=0,
            upper_bound=False,
            xhat=None,
            method="extensive_form",
        )

        print(f"Objective Value: {obj_value:.2f}")
        print("First-Stage Solution:")
        for var, value in first_stage_sol.items():
            print(f"  {var}: {value:.2f}")

    else:
        obj_val_SAA_list = []
        first_stage_sol_SAA_list = []

        if not upper_bound:
            for replication in range(1, replications + 1):
                start_rep = time.time()

                obj_value_SAA, first_stage_sol_SAA, up_bound = extensive_form(
                    d,
                    sampling,
                    iterations,
                    replication,
                    upper_bound,
                    xhat=None,
                    method="extensive_form",
                )
                obj_val_SAA_list.append(obj_value_SAA)
                first_stage_sol_SAA_list.append(first_stage_sol_SAA)

                end_rep = time.time()
                rep_time = end_rep - start_rep
                if replication % 1 == 0:
                    print(
                        f"Replication {replication} completed in {rep_time:.2f} seconds."
                    )

        else:
            # "Extra" replication for obtaining near optimal x_hat to compute upper bound
            obj_value, x_hat, up_bound = extensive_form(
                d,
                sampling,
                iterations,
                replication=0,
                upper_bound=False,
                xhat=None,
                method="extensive_form",
            )
            x_hat = list(x_hat.values())

            up_bound_list = []
            for replication in range(1, replications + 1):
                start_rep = time.time()

                obj_value_SAA, first_stage_sol_SAA, up_bound = extensive_form(
                    d, sampling, iterations, replication, upper_bound, x_hat, method
                )
                obj_val_SAA_list.append(obj_value_SAA)
                first_stage_sol_SAA_list.append(first_stage_sol_SAA)
                up_bound_list.append(up_bound)

                end_rep = time.time()
                rep_time = end_rep - start_rep
                if replication % 1 == 0:
                    print(
                        f"Replication {replication} completed in {rep_time:.2f} seconds."
                    )

            up_bound, up_bound_conf_interval = confidence_interval(up_bound_list)
            print(f"Upper Bound Estimate: {up_bound:.2f}")
            if up_bound_conf_interval is not None:
                print(
                    f"  Confidence Interval: [{up_bound_conf_interval[0]:.2f}, {up_bound_conf_interval[1]:.2f}]"
                )
            else:
                print("  Confidence Interval: None")

        lo_bound, lo_bound_conf_interval = confidence_interval(obj_val_SAA_list)

        print(f"Lower Bound Estimate: {lo_bound:.2f}")
        if lo_bound_conf_interval is not None:
            print(
                f"  Confidence Interval: [{lo_bound_conf_interval[0]:.2f}, {lo_bound_conf_interval[1]:.2f}]"
            )
        else:
            print("  Confidence Interval: None")

        averages = {}
        for first_stage_sol_SAA in first_stage_sol_SAA_list[0].keys():
            avg_value = sum(
                rep[first_stage_sol_SAA] for rep in first_stage_sol_SAA_list
            ) / len(first_stage_sol_SAA_list)
            averages[first_stage_sol_SAA] = avg_value

        print("Average first-stage solution:")
        for first_stage_sol_SAA, avg in averages.items():
            print(f"  {first_stage_sol_SAA}: {avg:.2f}")

    end_main = time.time()  # Record the end time
    total_time = end_main - start_main  # Calculate the elapsed time
    print(f"Total time for main: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
