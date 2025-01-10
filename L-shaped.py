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

        print("The number of iterations equals:", len(observations))
    return observations, probabilities


def add_optimality_cut(model, Beta, beta, eta, iteration):
    cut_expr = gb.LinExpr()
    cut_expr.addTerms(
        Beta, model.prob.master_vars[:-1]
    )  # Assuming the last variable is eta
    cut_expr.addTerms(1.0, eta)

    cut_name = f"optimality_cut_{iteration}"
    model.prob.master_model.addLConstr(cut_expr, gb.GRB.GREATER_EQUAL, beta, cut_name)
    # print(f"Added optimality cut: {cut_expr} >= {beta}")


def add_feasibility_cut(model, Gamma, gamma, iteration):
    cut_expr = gb.LinExpr()
    cut_expr.addTerms(Gamma, model.prob.master_vars[:-1])

    cut_name = f"feasibility_cut_{iteration}"
    model.prob.master_model.addLConstr(cut_expr, gb.GRB.GREATER_EQUAL, gamma, cut_name)
    # print(f"Added feasibility cut: {cut_expr} >= {gamma}")


def l_shaped(model, sampling, iterations, T_mat, instance, method="L-shaped"):
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    model.create_master(method)

    # Optionally save file as 'master.lp' in output directory
    os.makedirs(output_dir, exist_ok=True)
    master_model = os.path.join(output_dir, "master_0.lp")
    model.prob.master_model.write(master_model)

    model.prob.master_model.setParam("OutputFlag", 0)
    model.prob.master_model.optimize()
    cost_coeff = [var.getAttr("Obj") for var in model.prob.master_vars][:-1]

    # Obtain first-stage solution of master problem or initialize with zeroes
    if all(hasattr(var, "X") for var in model.prob.master_vars[:-1]):
        incmb = [var.X for var in model.prob.master_vars[:-1]]
    else:
        incmb = [0] * (len(model.prob.master_vars) - 1)

    print("First solution for x is", incmb)

    rand_vars = RandVars(model.name)
    observations, probabilities = get_obs_probs(rand_vars, sampling, iterations)

    convergence_criterion = False
    nu = 1
    lo_bound = -np.infty
    up_bound = np.infty
    epsilon = 10e-6

    # Create list of h_vec (for incmb is the zero solution)
    incmb_zero = [0] * (len(model.prob.master_vars) - 1)
    h_vec_list = []
    for obs in observations:
        model.create_LSsub(obs, incmb_zero, iteration=0)
        h_vec = np.array([const.RHS for const in model.prob.sub_model.getConstrs()])
        h_vec_list.append(h_vec)

    while not convergence_criterion:
        print("nu=", nu)

        # Step 1: Check if x is in K2
        for i, obs in enumerate(observations):
            model.create_feas_sub(obs, incmb, i)
            model.prob.feas_sub_model.setParam("OutputFlag", 0)
            model.prob.feas_sub_model.optimize()
            feas_obj_value = model.prob.feas_sub_model.objVal
            feas_sub_model = os.path.join(output_dir, f"feas_sub_{i}.lp")
            model.prob.feas_sub_model.write(feas_sub_model)

            if feas_obj_value > 0:
                sigma = np.array(
                    model.prob.feas_sub_model.getAttr(
                        "Pi", model.prob.feas_sub_model.getConstrs()
                    )
                )
                h_vec = h_vec_list[i]

                Gamma = np.dot(sigma.T, T_mat)
                gamma = np.dot(sigma.T, h_vec)

                add_feasibility_cut(model, Gamma, gamma, nu)
                # Break and skip to step 3
                break

        else:
            # Step 2: Compute an optimality cut
            Beta = 0
            beta = 0
            up_bound_sum = 0

            for i, obs in enumerate(observations):
                model.create_LSsub(obs, incmb, i)
                model.prob.sub_model.setParam("OutputFlag", 0)
                model.prob.sub_model.optimize()
                obj_value = model.prob.sub_model.objVal
                # sub_model = os.path.join(output_dir, f"sub_{i}.lp")
                # model.prob.sub_model.write(sub_model)

                dual_mult = np.array(
                    model.prob.sub_model.getAttr(
                        "Pi", model.prob.sub_model.getConstrs()
                    )
                )
                h_vec = h_vec_list[i]

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

            eta_var = model.prob.master_vars[-1]
            add_optimality_cut(model, Beta, beta, eta_var, nu)

        # Step 3: solve master program
        model.prob.master_model.optimize()
        master_model = os.path.join(output_dir, f"master_{nu}.lp")
        model.prob.master_model.write(master_model)

        incmb = [var.X for var in model.prob.master_vars][:-1]
        master_obj_value = model.prob.master_model.objVal

        new_lo_bound = master_obj_value
        if new_lo_bound >= lo_bound:
            lo_bound = new_lo_bound

        # Step 4: termination
        if abs(up_bound - lo_bound) < abs(epsilon * up_bound):
            convergence_criterion = True
            print("Solution of first-stage variables:")
            for var, value in zip(model.prob.master_vars, x_opt):
                print(f"    {var.varName}= {value:.2f}")
            print(f"Objective Value: {round(master_obj_value, 2)}")

        else:
            nu += 1


def main():
    start_main = time.time()

    instance = "Test_p214"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    sampling = False

    # Set parameters for the case sampling = True
    iterations = 200

    d = decompose(f"{instance}", input_dir)
    d.find_stage_idx()
    T_mat = get_T_mat(d)
    method = "L-shaped"

    l_shaped(d, sampling, iterations, T_mat, instance, method="L-shaped")

    end_main = time.time()  # Record the end time
    total_time = end_main - start_main  # Calculate the elapsed time
    print(f"Total time for main: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
