import os
import time
import sys

import itertools
import numpy as np
import random
import gurobipy as gb


# Get the directory containing readSMPS
readsmps_dir = "/Users/jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS"

# Add it to sys.path
if readsmps_dir not in sys.path:
    sys.path.append(readsmps_dir)

from readSMPS.decmps_2slp import RandVars
from readSMPS.decmps_2slp import decompose


def get_obs_probs(rvs, sampling=False, N=None):
    print("The number of random variables equals:", rvs.rvnum)
    if rvs.rvnum != len(rvs.dist):
        raise ValueError(
            "The number of random variables (rv) must match the number random variables in the dictionary."
        )

    if not sampling:
        # After processing the stochastic data, compute observations and probabilities
        rand_vars_values = [list(rvs.dist[i].keys()) for i in range(len(rvs.rv))]
        observations = list(itertools.product(*rand_vars_values))

        rand_vars_probs = [list(rvs.dist[i].values()) for i in range(len(rvs.rv))]
        combinations = itertools.product(*rand_vars_probs)
        probabilities = [np.prod(combination) for combination in combinations]

        print("The number of scenarios equals:", len(observations))

    else:
        if N is None:
            try:
                N = int(
                    input("Please enter the number of samples (N): ")
                )  # Get N from the user
                if N <= 0:
                    raise ValueError("N must be a positive integer.")
            except ValueError as e:
                raise ValueError(
                    f"Invalid input for N: {e}"
                )  # Raise an error if input is invalid

        observations = []
        for _ in range(N):
            sample_tuple = []
            for rv in rvs.dist:
                outcomes, probs = zip(*rv.items())  # Extract outcomes and probabilities
                sampled_outcome = random.choices(outcomes, probs)[
                    0
                ]  # Sample based on probabilities
                sample_tuple.append(sampled_outcome)
            observations.append(tuple(sample_tuple))

        probabilities = [1 / N] * N  # Equal probability for each sampled observation

    return observations, probabilities


def create_ef(model, observations, probabilities):
    for i, obs in enumerate(observations, start=1):
        prob = probabilities[i - 1]
        model.create_sub(obs, prob, i)


def main():
    start_main = time.time()

    instance = "20"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    sampling = True
    N = 50
    replications = 5

    for j in range(1, replications + 1):
        start_rep = time.time()

        print("Replication:", j)

        d = decompose(f"{instance}", input_dir)
        d.find_stage_idx()
        d.create_master(j)

        rand_vars = RandVars(d.name)
        observations, probabilities = get_obs_probs(rand_vars, sampling, N)

        create_ef(d, observations, probabilities)

        # Optionally save file as 'extensive_form_{j}.lp' in output directory
        os.makedirs(output_dir, exist_ok=True)
        extensive_form_file = os.path.join(output_dir, f"extensive_form_{j}.lp")
        d.prob.extensive_form.write(extensive_form_file)

        # Solve the extensive form
        d.prob.extensive_form.setParam("OutputFlag", 0)
        d.prob.extensive_form.optimize()

        obj_value = d.prob.extensive_form.ObjVal
        print(f"Objective Value: {obj_value}")

        # Retrieve first-stage variable values
        for v in d.prob.extensive_form.getVars()[: d.prob.master_var_size]:
            print(f"{v.VarName} = {v.X}")

        end_rep = time.time()
        rep_time = end_rep - start_rep
        print(f"Replication {j} completed in {rep_time:.2f} seconds.")

    end_main = time.time()  # Record the end time
    total_time = end_main - start_main  # Calculate the elapsed time
    print(f"Total time for main: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
