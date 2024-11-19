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

# from readSMPS.decmps_2slp import RandVars
from readSMPS.decmps_2slp import decompose
from itertools import product


def main():
    start = time.time()

    instance = "lands2"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    d = decompose(f"{instance}", input_dir)

    d.find_stage_idx()
    d.create_master()

    d.prob.master_model.write("mmmm.lp")
    print("msc vars",d.prob.master_vars)
    print("msc constraints",d.prob.master_const)

    d.create_sub()

    d.prob.sub_model.write("hhhhh.lp")
    print("sub vars",d.prob.sub_vars)
    print("sub constraints",d.prob.sub_const)



    # d.create_master()

    # print("master const",d.prob.master_const)
    # print("master vars",d.prob.master_vars)

    # Assuming 'd.prob.master_vars' contains the list of variables
    # and 'd.prob.master_const' contains the list of constraints


    # # Create master
    # d = create_master(instance, input_dir)

    # # Save the file in the newly created folder
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "test.lp")
    # d.prob.master_model.write(output_file)

    # d.prob.sub_model.write("test0.lp")

    # rand_vars = RandVars(d.name)

    # rand_vars_values = [list(rand_vars.dist[i].keys()) for i in range(len(rand_vars.rv))]
    # observations = list(product(*rand_vars_values))

    # print("observations", observations)

    # rand_vars_probs = [list(rand_vars.dist[i].values()) for i in range(len(rand_vars.rv))]
    # combinations = itertools.product(*rand_vars_probs)
    # probabilities = [np.prod(combination) for combination in combinations]
    
    # print("probabilities", probabilities)

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