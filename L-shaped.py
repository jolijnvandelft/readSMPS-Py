import gurobipy as gb
import numpy as np
import os
import time
import sys

# Get the directory containing readSMPS
readsmps_dir = "/Users/jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS"

# Add it to sys.path
if readsmps_dir not in sys.path:
    sys.path.append(readsmps_dir)

from readSMPS.decmps_2slp import RandVars
from readSMPS.create_master import create_master
from itertools import product

def solve_master(d):
    d.prob.master_model.setParam("OutputFlag", 0)
    d.prob.master_model.optimize()

    if d.prob.master_model.status == gb.GRB.OPTIMAL:
        objective_value = d.prob.master_model.objVal
        print("objective_value", objective_value)

        incmb = []
        cost_vector = []

        for var in d.prob.master_vars:
            if 'eta' not in var.varName:
                incmb.append(var.x)
                cost_vector.append(var.obj)
            #print("vars",var)

        return incmb, cost_vector, objective_value
    else:
        print("Master problem did not solve to optimality. Status code:", d.prob.master_model.status)

        return None
    
def solve_subproblem(d, obs, incmb, iteration, i):
    d.create_LSsub(obs, incmb)
    subproblem_filename = f"iteration_{iteration}_sub_problem_{i+1}.lp"
    #d.prob.sub_model.write(subproblem_filename)
    
    d.prob.sub_model.setParam("OutputFlag", 0)
    d.prob.sub_model.optimize()

    if d.prob.sub_model.status == gb.GRB.OPTIMAL:
        dual_multipliers = d.prob.sub_model.getAttr('Pi', d.prob.sub_model.getConstrs())
        objective_value = d.prob.sub_model.objVal

        return dual_multipliers, objective_value
    else:
        print(f"Subproblem {i+1} did not solve to optimality. Status code:", d.prob.sub_model.status)

        return None, None
    
def add_optimality_cut(d, beta, beta_0, eta, iteration):
    cut_expr = gb.LinExpr()
    cut_expr.addTerms(beta, d.prob.master_vars[:-1])  # Assuming the last variable is eta
    cut_expr.addTerms(1.0, eta)

    cut_name = f"cut_{iteration}"
    d.prob.master_model.addLConstr(cut_expr, gb.GRB.GREATER_EQUAL, beta_0, cut_name)

def create_T_mat(d):
    T_mat_vars = d.prob.mean_model.getVars()[:d.tim.stage_idx_col[1]]
    T_mat_const = d.prob.mean_model.getConstrs()[d.tim.stage_idx_row[1]:]

    T_mat = [[0 for _ in range(len(T_mat_vars))] for _ in range(len(T_mat_const))]

    for i, const in enumerate(T_mat_const):
        for j, var in enumerate(T_mat_vars):
            coeff = d.prob.mean_model.getCoeff(const, var)
            T_mat[i][j] = coeff

    return T_mat

def create_h_vec(d, obs, incmb):
    d.create_LSsub(obs, incmb)
    h_vec = np.array([const.RHS for const in d.prob.sub_model.getConstrs()])

    return h_vec

def calculate_probabilities(rand_vars):
    keys_list = [list(rand_vars.dist[i].keys()) for i in range(len(rand_vars.dist))]
    obs_list = list(product(*keys_list))
    probabilities = [list(rand_vars.dist[i].values()) for i in range(len(rand_vars.dist))]

    probs_scen_list = []
    for scenario in obs_list:
        prob = 1
        for i, value in enumerate(scenario):
            prob_index = keys_list[i].index(value)
            prob *= probabilities[i][prob_index]
        probs_scen_list.append(prob)

    return probs_scen_list

def main():
    start = time.time()

    instance = "lands"
    input_dir = "/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Input/"
    output_dir = f"/Users/Jolijn/Documents/Berlin/Thesis/Code/readSMPS-Py/readSMPS/Output/{instance}"

    # Create model
    d = create_master(instance, input_dir)

    # Save the file in the newly created folder
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "Master_0.lp")
    d.prob.master_model.write(output_file)
    
    # Initialize the RandVars class to read scenario data
    rand_vars = RandVars(d.name)
    
    rand_vars_values = [list(rand_vars.dist[i].keys()) for i in range(len(rand_vars.dist))]
    obs_list = list(product(*rand_vars_values))
    probs_scen_list = calculate_probabilities(rand_vars)

    #Initialize for the while-loop
    convergence_criterion = False
    iteration=1
    incmb, cost_vector, obj_value = solve_master(d)
    lo_bound = -gb.GRB.INFINITY
    up_bound = gb.GRB.INFINITY
    epsilon = 10e-6

    T_mat = np.array(create_T_mat(d))

    #For initialization of h create the zero solution
    incmb_zero = [0]*(len(d.prob.master_vars)-1)

    while not convergence_criterion: # and iteration < 2:
        print("iteration", iteration)
        beta = 0
        beta_0 = 0
        upper_bound_sum = 0

        for i, obs in enumerate(obs_list):
            h_vec = create_h_vec(d, obs, incmb_zero)
            dual_multipliers, objective_value = solve_subproblem(d, obs, incmb, iteration, i)

            # Compute beta_0 and beta
            if dual_multipliers is not None:
                prob_scen = probs_scen_list[i]
                pi_vec = np.array(dual_multipliers)

                beta_sum = np.dot(pi_vec.T, T_mat)
                beta_0_sum = np.dot(pi_vec, h_vec)

            beta += prob_scen * beta_sum
            beta_0 += prob_scen * beta_0_sum
            upper_bound_sum += prob_scen * objective_value

        # Compute upper bound and update up_bound
        upper_bound = np.dot(np.array(cost_vector), np.array(incmb)) + upper_bound_sum
        
        if upper_bound <= up_bound:
            up_bound = upper_bound

        eta_var = d.prob.master_vars[-1]

        #Add optimality cut
        add_optimality_cut(d,beta,beta_0,eta_var,iteration)

        #Solve new master problem
        incmb, cost_vector, objective_master_value = solve_master(d)

        output_file = os.path.join(output_dir, f"Master_{iteration}.lp")
        d.prob.master_model.write(output_file)

        if objective_master_value >= lo_bound:
            lo_bound = objective_master_value

        if abs(up_bound - lo_bound) < epsilon:
            convergence_criterion = True
            end = time.time()
            print("")
            print(f"Convergence criterion met in {round(end-start,2)} seconds")

            print("Master variables and their values:")
            for var in d.prob.master_vars:
                print(f"{var.varName}: {round(var.x,2)}")

            obj_value = d.prob.master_model.ObjVal
            print(f"Objective Value: {round(obj_value,2)}")
        else:
            iteration +=1

if __name__ == "__main__":
    main()