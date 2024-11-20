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

    instance = "lands"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    d = decompose(f"{instance}", input_dir)

    d.find_stage_idx()
    d.create_master()

    os.makedirs(output_dir, exist_ok=True)
    master_file = os.path.join(output_dir, "master.lp")
    d.prob.master_model.write(master_file)

    # print("msc vars", d.prob.master_vars)
    # print("msc constraints", d.prob.master_const)

    # Instantiate RandVars
    rand_vars = RandVars(d.name)

    for i, obs in enumerate(rand_vars.observations, start=1):
        d.create_sub(obs, i)
        sub_file = os.path.join(output_dir, f"sub_{i}.lp")
        d.prob.sub_model.write(sub_file)

    # print("sub vars", d.prob.sub_vars)
    # print("sub vars stage 2", d.prob.sub_vars_s2)
    # print("sub constraints", d.prob.sub_const)

    # incmb_zero = [0]*(len(d.prob.master_vars))
    # print("incmb_zero", incmb_zero)

    # d.create_LSsub(observations[0], incmb_zero)

    # test_sub = "sub_0.lp"
    # d.prob.sub_model.write(test_sub)

    # d.prob.sub_model.optimize()

    # if d.prob.sub_model.status == gb.GRB.OPTIMAL:
    #     print("YES")


if __name__ == "__main__":
    main()
