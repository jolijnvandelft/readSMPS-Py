import gurobipy as gb
import numpy as np
import os
import time
import sys
import itertools

# Get the directory containing readSMPS
readsmps_dir = "/Users/jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS"

# Add it to sys.path
if readsmps_dir not in sys.path:
    sys.path.append(readsmps_dir)

from readSMPS.decmps_2slp import RandVars
from readSMPS.decmps_2slp import decompose


def main():
    start = time.time()

    instance = "lands2"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    d = decompose(f"{instance}", input_dir)

    d.find_stage_idx()
    d.create_master()

    os.makedirs(output_dir, exist_ok=True)
    master_file = os.path.join(output_dir, "master.lp")
    d.prob.master_model.write(master_file)

    rand_vars = RandVars(d.name)

    for i, obs in enumerate(rand_vars.observations, start=1):
        prob = rand_vars.probabilities[i - 1]
        d.create_sub(obs, prob, i)
        sub_file = os.path.join(output_dir, f"sub_{i}.lp")
        d.prob.sub_model.write(sub_file)


if __name__ == "__main__":
    main()
