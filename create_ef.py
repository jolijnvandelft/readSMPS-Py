import os
import time
import sys

import itertools
import numpy as np
import random

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


def main():
    start = time.time()

    instance = "storm"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    d = decompose(f"{instance}", input_dir)

    d.find_stage_idx()
    d.create_master()

    rand_vars = RandVars(d.name)

    observations, probabilities = get_obs_probs(rand_vars, sampling=True, N=100)

    for i, obs in enumerate(observations, start=1):
        prob = probabilities[i - 1]
        d.create_sub(obs, prob, i)

    os.makedirs(output_dir, exist_ok=True)
    extensive_form_file = os.path.join(output_dir, "extensive_form.lp")
    # d.prob.extensive_form.write(extensive_form_file)

    d.prob.extensive_form.setParam("OutputFlag", 0)
    d.prob.extensive_form.optimize()

    end = time.time()  # Record the end time
    elapsed_time = end - start  # Calculate the elapsed time
    print(f"Solved in {elapsed_time:.2f} seconds")  # Print the elapsed time

    obj_value = d.prob.extensive_form.ObjVal
    print(f"Objective Value: {obj_value}")

    # Retrieve first-stage variable values
    for v in d.prob.extensive_form.getVars()[: d.prob.master_var_size]:
        print(f"{v.VarName} = {v.X}")


if __name__ == "__main__":
    main()
