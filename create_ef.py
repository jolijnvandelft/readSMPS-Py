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


def create_ef(model, sampling=False, iterations=None, replication=0):
    rand_vars = RandVars(model.name)
    observations, probabilities = get_obs_probs(rand_vars, sampling, iterations)

    model.create_master(replication)

    for iteration, obs in enumerate(observations, start=1):
        prob = probabilities[iteration - 1]
        model.create_sub(obs, prob, iteration)

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

    return obj_value, first_stage_sol


def lower_bound(lo_bnd_list, alpha=0.05):
    M = len(lo_bnd_list)

    mean = sum(lo_bnd_list) / len(lo_bnd_list)
    diff = [(v - mean) for v in lo_bnd_list]
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

    instance = "lands2"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    sampling = True

    # Set parameters for the case sampling = True
    iterations = 20
    replications = 100
    bounds = True

    d = decompose(f"{instance}", input_dir)
    d.find_stage_idx()

    if not sampling:
        obj_value, first_stage_sol = create_ef(
            d, sampling, iterations=None, replication=0
        )

        print("Objective Value:", obj_value)
        print("First-stage solution:", first_stage_sol)

    else:
        objective_values = []

        for replication in range(replications):
            start_rep = time.time()

            obj_value, first_stage_sol = create_ef(d, sampling, iterations, replication)
            objective_values.append(obj_value)

            # print("Replication:", replication + 1)
            # print("Objective Value:", obj_value)
            # print("First-stage solution:", first_stage_sol)

            end_rep = time.time()
            rep_time = end_rep - start_rep
            print(f"Replication {replication+1} completed in {rep_time:.2f} seconds.")

        if bounds == True:
            lo_bound, lo_bound_conf_interval = lower_bound(objective_values)

            print("Lower bound estimate:", lo_bound)
            print("Lower bound confidence interval:", lo_bound_conf_interval)

    end_main = time.time()  # Record the end time
    total_time = end_main - start_main  # Calculate the elapsed time
    print(f"Total time for main: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
