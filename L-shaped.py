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


def main():
    start_main = time.time()

    instance = "lands2"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    sampling = False
    iterations = None

    d = decompose(f"{instance}", input_dir)
    d.find_stage_idx()
    d.create_master()

    # Optionally save file as 'master.lp' in output directory
    os.makedirs(output_dir, exist_ok=True)
    master_model = os.path.join(output_dir, "master.lp")
    d.prob.master_model.write(master_model)

    rand_vars = RandVars(d.name)
    observations, probabilities = get_obs_probs(rand_vars, sampling, iterations)

    print("observations", observations)
    print("probs", probabilities)

    x_opt = [-4, -8, 8, 10]  # for testing

    for i, obs in enumerate(observations):
        d.create_LSsub(obs, x_opt)
        sub_model = os.path.join(output_dir, f"sub_{i}.lp")
        d.prob.sub_model.write(sub_model)

    end_main = time.time()  # Record the end time
    total_time = end_main - start_main  # Calculate the elapsed time
    print(f"Total time for main: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
