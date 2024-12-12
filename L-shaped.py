import os
import time
import sys
import gurobipy as gb

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


def get_T_mat(model):
    T_mat_vars = model.prob.mean_model.getVars()[: model.tim.stage_idx_col[1]]
    T_mat_const = model.prob.mean_model.getConstrs()[model.tim.stage_idx_row[1] :]

    T_mat = [
        [model.prob.mean_model.getCoeff(const, var) for var in T_mat_vars]
        for const in T_mat_const
    ]

    return T_mat


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


def add_optimality_cut(d, Beta, beta, eta, iteration):
    cut_expr = gb.LinExpr()
    cut_expr.addTerms(
        Beta, d.prob.master_vars[:-1]
    )  # Assuming the last variable is eta
    cut_expr.addTerms(1.0, eta)

    cut_name = f"cut_{iteration}"
    d.prob.master_model.addLConstr(cut_expr, gb.GRB.GREATER_EQUAL, beta, cut_name)


def main():
    start_main = time.time()

    instance = "lands"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    sampling = False
    iterations = 100

    d = decompose(f"{instance}", input_dir)
    d.find_stage_idx()
    T_mat = get_T_mat(model=d)

    # START of L-shaped method

    d.create_master()

    # Optionally save file as 'master.lp' in output directory
    os.makedirs(output_dir, exist_ok=True)
    master_model = os.path.join(output_dir, "master_0.lp")
    d.prob.master_model.write(master_model)

    d.prob.master_model.setParam("OutputFlag", 0)
    d.prob.master_model.optimize()
    cost_coeff = [var.getAttr("Obj") for var in d.prob.master_vars][:-1]
    incmb = [var.X for var in d.prob.master_vars][:-1]

    rand_vars = RandVars(d.name)
    observations, probabilities = get_obs_probs(rand_vars, sampling, iterations)

    convergence_criterion = False
    k = 1
    lo_bound = -np.infty
    up_bound = np.infty
    epsilon = 10e-6

    while not convergence_criterion and k < 6:
        # Step 1: Check if x is in K2

        # Step 2: Compute an optimality cut
        Beta = 0
        beta = 0
        up_bound_sum = 0

        for i, obs in enumerate(observations):
            d.create_LSsub(obs, incmb)
            d.prob.sub_model.setParam("OutputFlag", 0)
            d.prob.sub_model.optimize()
            obj_value = d.prob.sub_model.objVal
            # sub_model = os.path.join(output_dir, f"sub_{i}.lp")
            sub_const = d.prob.sub_model.getConstrs()
            h_vec = np.array(d.prob.sub_model.getAttr("RHS", sub_const))
            dual_mult = np.array(d.prob.sub_model.getAttr("Pi", sub_const))

            prob = probabilities[i]
            Beta_sum = np.dot(dual_mult.T, T_mat)
            beta_sum = np.dot(dual_mult.T, h_vec)

            Beta += prob * Beta_sum
            beta += prob * beta_sum
            up_bound_sum += prob * obj_value

        new_up_bound = np.dot(np.array(cost_coeff), np.array(incmb)) + up_bound_sum

        if new_up_bound <= up_bound:
            up_bound = new_up_bound
            x_opt = incmb

        eta_var = d.prob.master_vars[-1]
        add_optimality_cut(d, Beta, beta, eta_var, k)

        # Step 3: solve master program
        d.prob.master_model.optimize()
        master_model = os.path.join(output_dir, f"master_{k}.lp")
        d.prob.master_model.write(master_model)
        
        incmb = [var.X for var in d.prob.master_vars][:-1]
        master_obj_value = d.prob.master_model.objVal

        new_lo_bound = master_obj_value
        if new_lo_bound >= lo_bound:
            lo_bound = new_lo_bound

        print("up_bound", up_bound)
        print("lo_bound", lo_bound)

        if abs(up_bound - lo_bound) < epsilon:
            convergence_criterion = True
            print(f"Optimal solution: {x_opt}")
            print(f"Objective Value: {round(obj_value, 2)}")

        else:
            k+=1

    end_main = time.time()  # Record the end time
    total_time = end_main - start_main  # Calculate the elapsed time
    print(f"Total time for main: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
