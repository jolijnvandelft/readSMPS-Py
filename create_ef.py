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
    if not sampling:
        # After processing the stochastic data, compute observations and probabilities
        rand_vars_values = [list(rvs.dist[i].keys()) for i in range(len(rvs.rv))]
        observations = list(itertools.product(*rand_vars_values))

        rand_vars_probs = [list(rvs.dist[i].values()) for i in range(len(rvs.rv))]
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


def create_ef(
    model, sampling=False, iterations=None, replication=0, upper_bound=False, xhat=None
):
    rand_vars = RandVars(model.name)
    observations, probabilities = get_obs_probs(rand_vars, sampling, iterations)

    model.create_master(replication)

    if not upper_bound:
        for iteration, obs in enumerate(observations, start=1):
            prob = probabilities[iteration - 1]
            model.create_sub(obs, prob, iteration)
        up_bound = None

    if upper_bound and replication > 0:
        variables = model.prob.extensive_form.getVars()
        cost_coeff = [var.getAttr("Obj") for var in variables]

        recourse_val_list = []
        for iteration, obs in enumerate(observations, start=1):
            prob = probabilities[iteration - 1]
            model.create_sub(obs, prob, iteration)

            model.create_LSsub(obs, xhat)
            model.prob.sub_model.setParam("OutputFlag", 0)
            model.prob.sub_model.optimize()
            recourse_val = model.prob.sub_model.objVal
            recourse_val_list.append(recourse_val)

        mean_recourse_val = sum(recourse_val_list) / len(recourse_val_list)

        up_bound = np.dot(cost_coeff, xhat) + mean_recourse_val

    # # Optionally save file as 'extensive_form.lp' in output directory
    # os.makedirs(output_dir, exist_ok=True)
    # extensive_form_file = os.path.join(output_dir, f"extensive_form_{replication}.lp")
    # d.prob.extensive_form.write(extensive_form_file)

    # Solve the extensive form
    model.prob.extensive_form.setParam("OutputFlag", 0)
    model.prob.extensive_form.optimize()

    obj_value = model.prob.extensive_form.ObjVal

    # Retrieve first-stage variable values
    first_stage_sol = {}
    for v in model.prob.extensive_form.getVars()[: model.prob.master_var_size]:
        first_stage_sol[v.VarName] = v.X

    return obj_value, first_stage_sol, up_bound


def confidence_interval(value_list, alpha=0.05):
    M = len(value_list)

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
    iterations = 50
    replications = 1000
    upper_bound = True

    d = decompose(f"{instance}", input_dir)
    d.find_stage_idx()

    if not sampling:
        obj_value, first_stage_sol, up_bound = create_ef(
            d, sampling, iterations=None, replication=0, upper_bound=False, xhat=None
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

                obj_value_SAA, first_stage_sol_SAA, up_bound = create_ef(
                    d, sampling, iterations, replication, upper_bound, xhat=None
                )
                obj_val_SAA_list.append(obj_value_SAA)
                first_stage_sol_SAA_list.append(first_stage_sol_SAA)

                end_rep = time.time()
                rep_time = end_rep - start_rep
                if replication % 50 == 0:
                    print(f"Replication {replication} completed in {rep_time:.2f} seconds.")

        else:
            # "Extra" replication for obtaining near optimal x_hat to compute upper bound
            obj_value, x_hat, up_bound = create_ef(
                d, sampling, iterations, replication=0, upper_bound=False, xhat=None
            )
            x_hat = list(x_hat.values())

            up_bound_list = []
            for replication in range(1, replications + 1):
                start_rep = time.time()

                obj_value_SAA, first_stage_sol_SAA, up_bound = create_ef(
                    d, sampling, iterations, replication, upper_bound, x_hat
                )
                obj_val_SAA_list.append(obj_value_SAA)
                first_stage_sol_SAA_list.append(first_stage_sol_SAA)
                up_bound_list.append(up_bound)

                end_rep = time.time()
                rep_time = end_rep - start_rep
                if replication % 50 == 0:
                    print(f"Replication {replication} completed in {rep_time:.2f} seconds.")

            up_bound, up_bound_conf_interval = confidence_interval(up_bound_list)
            print(f"Upper Bound Estimate: {up_bound:.2f}")
            print(f"  Confidence Interval: [{up_bound_conf_interval[0]:.2f}, {up_bound_conf_interval[1]:.2f}]")


        lo_bound, lo_bound_conf_interval = confidence_interval(obj_val_SAA_list)

        print(f"Lower Bound Estimate: {lo_bound:.2f}")
        print(f"  Confidence Interval: [{lo_bound_conf_interval[0]:.2f}, {lo_bound_conf_interval[1]:.2f}]")

        averages = {}
        for key in first_stage_sol_SAA_list[0].keys():  # Loop through the keys in the first dictionary
            # Calculate the average value for this key across all replications
            avg_value = sum(rep[key] for rep in first_stage_sol_SAA_list) / len(first_stage_sol_SAA_list)
            averages[key] = avg_value
       
        print("Average first-stage solution:")
        for key, avg in averages.items():
            print(f"  {key}: {avg:.2f}")

    end_main = time.time()  # Record the end time
    total_time = end_main - start_main  # Calculate the elapsed time
    print(f"Total time for main: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
